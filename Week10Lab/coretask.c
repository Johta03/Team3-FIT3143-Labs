#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>

// MPI Cartesian Topology Constants
#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1
#define NO_NEIGHBOR -2

// Message Tags
#define MSG_REPORT 1                        // Invader status update to master
#define MSG_DISABLE 2                       // Invader killed notification
#define MSG_TERMINATE 3                     // Game over signal
#define MSG_CANNONBALL_FROM_INVADER 4       // Invader fires at player

// Game Constants
#define MAX_CANNONBALLS 100

// Cannonball Tracking Structure
typedef struct {
    int col;              // Column position
    int target_row;       // Target row (for player cannonballs)
    int time_to_hit;      // Countdown timer (seconds until impact)
    int is_active;        // 1 = flying, 0 = inactive
    int source;           // Source rank
} Cannonball;

// Global cannonball arrays (tracked by master)
Cannonball player_cannonballs[MAX_CANNONBALLS];
Cannonball invader_cannonballs[MAX_CANNONBALLS];

// Performance Metrics (tracked by master)
double total_tick_time = 0.0;
int total_player_shots = 0;
int total_invader_shots = 0;
int total_hits = 0;

// Function Prototypes
int master_io(MPI_Comm world_comm, MPI_Comm comm, int nrows, int ncols);
int slave_io(MPI_Comm world_comm, MPI_Comm comm, int nrows, int ncols);
void printGrid(int rows, int cols, int **ppData, int masterCol);
static inline int TwoDToOneD(int i, int j, int cols) { return (i * cols) + j; }

// MAIN - Process Splitting
int main(int argc, char **argv)
{
	int rank, size, nrows, ncols;
	MPI_Comm new_comm;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Parse command-line arguments for grid dimensions
	if (argc == 3) {
        nrows = atoi(argv[1]);
        ncols = atoi(argv[2]);
        if ((nrows * ncols) != (size-1)) {
            if (rank == 0)
                printf("ERROR: Grid %dx%d = %d invaders, but need %d processes total (got %d)\n", 
                       nrows, ncols, nrows*ncols, nrows*ncols+1, size);
			MPI_Finalize();
			return 0;
        }
    } else {
        nrows = ncols = 0;  // Auto-calculate square grid
    }
	
	srand(time(NULL) + rank);  // Unique random seed per process

	// Split into Master (rank == size-1) and Slaves (rank < size-1)
	MPI_Comm_split(MPI_COMM_WORLD, rank == (size-1), 0, &new_comm);
	
	if (rank == size-1) 
		master_io(MPI_COMM_WORLD, new_comm, nrows, ncols);
	else
		slave_io(MPI_COMM_WORLD, new_comm, nrows, ncols);
		
	MPI_Finalize();
	return 0;
}

// PRINT GRID - Display invader states and player position
void printGrid(int rows, int cols, int **ppData, int masterCol)
{
	int idx;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			idx = TwoDToOneD(i, j, cols);
			char status = (ppData[idx][0] == MSG_REPORT) ? 'A' :      // Active
			              (ppData[idx][0] == MSG_DISABLE) ? 'D' :      // Disabled
			              (ppData[idx][0] == MSG_TERMINATE) ? 'X' : '?'; // Unknown
			printf("(%d,%d)%c:%d\t", i, j, status, ppData[idx][1]);
		}
		printf("\n");
	}
	
	// Display player position
	for(int i = 0; i < cols; i++){
		printf("%s\t", (masterCol == i) ? "^PLAYER^" : "________");
	}
	printf("\n\n");
	fflush(stdout);
}

// MASTER PROCESS - Player Control & Game Coordination
int master_io(MPI_Comm world_comm, MPI_Comm comm, int nrows, int ncols)
{
	int global_rank, global_size, local_rank, local_size;
	int tick = 0, master_col = 0;
	int SlaveActiveCount, SlaveTerminateCount = 0;
	int **ppSlaveGrid;
	int terminateSignalSent = 0;
	
	MPI_Status status, probe_status;
	int flag = 0, recv_Val, send_Val;
	
	MPI_Comm_size(world_comm, &global_size);
	MPI_Comm_rank(world_comm, &global_rank);
	MPI_Comm_size(comm, &local_size);
	MPI_Comm_rank(comm, &local_rank);

	// Auto-calculate grid dimensions if not specified
	if(nrows == 0 && ncols == 0)
		nrows = ncols = (int)sqrt(global_size-1);
	
	SlaveActiveCount = nrows * ncols;
	
	// Allocate and initialize invader tracking grid
	ppSlaveGrid = (int**)malloc(SlaveActiveCount * sizeof(int*));
	for(int i = 0; i < SlaveActiveCount; i++){
		ppSlaveGrid[i] = (int*)malloc(2 * sizeof(int));
		ppSlaveGrid[i][0] = MSG_REPORT;  // All start active
		ppSlaveGrid[i][1] = 0;
	}
	
	// Initialize cannonball arrays
	for(int i = 0; i < MAX_CANNONBALLS; i++){
		player_cannonballs[i].is_active = 0;
		invader_cannonballs[i].is_active = 0;
	}
	
	// Display initial game state
	printf("========================================\n");
	printf("   SPACE INVADERS SIMULATION START\n");
	printf("========================================\n");
	printf("Grid: %d rows x %d columns = %d invaders\n", nrows, ncols, SlaveActiveCount);
	printf("Player starts at Column %d\n\n", master_col);
	printGrid(nrows, ncols, ppSlaveGrid, master_col);

	// === MAIN GAME LOOP ===
	while(1){
		tick++;
		
		// Broadcast tick to ALL processes for synchronization
		MPI_Bcast(&tick, 1, MPI_INT, global_rank, MPI_COMM_WORLD);

		// -Player Movement simulated
		// Strategy: Move toward column with closest (bottom-most) alive invader
		int best_col = -1, closest_row = -1;
		
		// Iterates through each column starting from the left
		for(int col = 0; col < ncols; col++) {
			// Iterates through each row starting from the bottom most
			for(int row = nrows-1; row >= 0; row--) {
				int idx = TwoDToOneD(row, col, ncols);
				if(ppSlaveGrid[idx][0] == MSG_REPORT) {  // Found alive invader

					if(row > closest_row // Is this invader CLOSER (lower down) than current best? Higher row number = closer to player
						|| 
					   (row == closest_row && abs(col - master_col) < abs(best_col - master_col))) 
					   // "If SAME distance, which column is closer to player?"
					   // abs(col - master_col): Distance from invader column to player column
					   // abs(best_col - master_col): Distance from current best to player
					   // Choose smallest distance
					   {
						closest_row = row;
						best_col = col;
					}
					break;
				}
			}
		}
		
		// Move one step toward target
		printf("[Tick %d] ", tick);
		if(best_col == -1) {
			printf("Player Col %d | No targets\n", master_col);
		} else if(best_col > master_col && master_col < ncols-1) {
			master_col++;
			printf("Player moves RIGHT → Col %d\n", master_col);
		} else if(best_col < master_col && master_col > 0) {
			master_col--;
			printf("Player moves LEFT ← Col %d\n", master_col);
		} else {
			printf("Player stays at Col %d\n", master_col);
		}
		
		// Player Fires
		// Auto-fire at bottom-most invader in current column
		int target_row = -1;
		for(int i = nrows-1; i >= 0; i--) {
			int idx = TwoDToOneD(i, master_col, ncols);
			if(ppSlaveGrid[idx][0] == MSG_REPORT) {
				target_row = i;
				break;
			}
		}
		
		if(target_row != -1) {
			int travel_time = 2 + (nrows - 1 - target_row);
			printf("          Fire → Invader(%d,%d) [ETA: %d sec]\n", target_row, master_col, travel_time);
			
			for(int i = 0; i < MAX_CANNONBALLS; i++) {
				if(!player_cannonballs[i].is_active) {
					player_cannonballs[i].col = master_col;
					player_cannonballs[i].target_row = target_row;
					player_cannonballs[i].time_to_hit = travel_time;
					player_cannonballs[i].is_active = 1;
					player_cannonballs[i].source = global_rank;
					break;
				}
			}
		}

		// --- RECEIVE MESSAGES FROM INVADERS ---
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, world_comm, &flag, &probe_status);
		if(flag){
			flag = 0;
			switch(probe_status.MPI_TAG)
			{
				case MSG_REPORT:
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_REPORT, world_comm, &status);
					if(ppSlaveGrid[probe_status.MPI_SOURCE][0] == MSG_REPORT) {
						ppSlaveGrid[probe_status.MPI_SOURCE][1] = recv_Val;
					}
					break;
					
				case MSG_DISABLE:
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_DISABLE, world_comm, &status);
					ppSlaveGrid[probe_status.MPI_SOURCE][0] = MSG_DISABLE;
					ppSlaveGrid[probe_status.MPI_SOURCE][1] = recv_Val;
					break;
					
				case MSG_CANNONBALL_FROM_INVADER: {
					int msg_data[2];
					MPI_Recv(msg_data, 2, MPI_INT, probe_status.MPI_SOURCE, MSG_CANNONBALL_FROM_INVADER, world_comm, &status);
					printf("          !!! Invader at col %d fires! ETA: %d sec\n", msg_data[1], msg_data[0]);
					
					for(int i = 0; i < MAX_CANNONBALLS; i++) {
						if(!invader_cannonballs[i].is_active) {
							invader_cannonballs[i].col = msg_data[1];
							invader_cannonballs[i].time_to_hit = msg_data[0];
							invader_cannonballs[i].is_active = 1;
							invader_cannonballs[i].source = probe_status.MPI_SOURCE;
							break;
						}
					}
					break;
				}
				
				case MSG_TERMINATE:
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_TERMINATE, world_comm, &status);
					ppSlaveGrid[probe_status.MPI_SOURCE][0] = MSG_TERMINATE;
					SlaveTerminateCount++;
					break;
					
				default:
					printf("Master: Invalid Tag received\n");
					break;
			}
		}
		
		// UPDATE PLAYER CANNONBALLS & CHECK HITS 
		// Extension (a): Deflection logic - 20% left, 15% right, 20% block, 45% hit
		for(int i = 0; i < MAX_CANNONBALLS; i++) {
			if(!player_cannonballs[i].is_active) continue;
			
			player_cannonballs[i].time_to_hit--;
			if(player_cannonballs[i].time_to_hit > 0) continue;
			
			// Cannonball reached target
			int trow = player_cannonballs[i].target_row;
			int tcol = player_cannonballs[i].col;
			int trank = TwoDToOneD(trow, tcol, ncols);
			
			if(ppSlaveGrid[trank][0] == MSG_REPORT) {  // Target still alive
				int deflect = rand() % 100;
				int kill_msg = 1;
				
				if(deflect < 20) {  // 20% deflect LEFT
					int lcol = tcol - 1;
					if(lcol >= 0 && ppSlaveGrid[TwoDToOneD(trow, lcol, ncols)][0] == MSG_REPORT) {
						printf("          <- DEFLECT LEFT! Invader(%d,%d) destroyed!\n", trow, lcol);
						MPI_Send(&kill_msg, 1, MPI_INT, TwoDToOneD(trow, lcol, ncols), MSG_DISABLE, world_comm);
						ppSlaveGrid[TwoDToOneD(trow, lcol, ncols)][0] = MSG_DISABLE;
						SlaveActiveCount--;
						total_hits++;
					} else {
						printf("          <- Deflected left (no target)\n");
					}
				} else if(deflect < 35) {  // 15% deflect RIGHT
					int rcol = tcol + 1;
					if(rcol < ncols && ppSlaveGrid[TwoDToOneD(trow, rcol, ncols)][0] == MSG_REPORT) {
						printf("          -> DEFLECT RIGHT! Invader(%d,%d) destroyed!\n", trow, rcol);
						MPI_Send(&kill_msg, 1, MPI_INT, TwoDToOneD(trow, rcol, ncols), MSG_DISABLE, world_comm);
						ppSlaveGrid[TwoDToOneD(trow, rcol, ncols)][0] = MSG_DISABLE;
						SlaveActiveCount--;
						total_hits++;
					} else {
						printf("          -> Deflected right (no target)\n");
					}
				} else if(deflect < 55) {  // 20% BLOCKED
					printf("          BLOCKED! Invader(%d,%d) shield holds!\n", trow, tcol);
				} else {  // 45% DIRECT HIT
					printf("          *** HIT! Invader(%d,%d) destroyed! ***\n", trow, tcol);
					MPI_Send(&kill_msg, 1, MPI_INT, trank, MSG_DISABLE, world_comm);
					ppSlaveGrid[trank][0] = MSG_DISABLE;
					SlaveActiveCount--;
					total_hits++;
				}
			}
			player_cannonballs[i].is_active = 0;
		}
		
		// UPDATE INVADER CANNONBALLS & CHECK PLAYER HIT 
		int player_hit = 0;
		for(int i = 0; i < MAX_CANNONBALLS; i++) {
			if(!invader_cannonballs[i].is_active) continue;
			
			invader_cannonballs[i].time_to_hit--;
			if(invader_cannonballs[i].time_to_hit > 0) continue;
			
			if(invader_cannonballs[i].col == master_col) {
				printf("\n========================================\n");
				printf("  PLAYER HIT! GAME OVER - YOU LOSE!\n");
				printf("========================================\n");
				player_hit = 1;
				terminateSignalSent = 1;
				
				// Send terminate to all slaves
				for(int j = 0; j < (global_size-1); j++){
					send_Val = MSG_TERMINATE;
					MPI_Send(&send_Val, 1, MPI_INT, j, MSG_TERMINATE, world_comm);
				}
			} else {
				printf("          Miss! Shot at col %d (player at %d)\n", invader_cannonballs[i].col, master_col);
			}
			invader_cannonballs[i].is_active = 0;
		}
		
		if(player_hit) {
			// Wait for all slaves to acknowledge termination
			while(SlaveTerminateCount < (global_size-1)) {
				MPI_Iprobe(MPI_ANY_SOURCE, MSG_TERMINATE, world_comm, &flag, &probe_status);
				if(flag) {
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_TERMINATE, world_comm, &status);
					ppSlaveGrid[probe_status.MPI_SOURCE][0] = MSG_TERMINATE;
					SlaveTerminateCount++;
				}
			}
			
			// Show final grid with all invaders terminated (X status)
			printf("\nFinal Grid State:\n");
			printGrid(nrows, ncols, ppSlaveGrid, master_col);

			break;  // Exit game loop
		}
		
		// --- PHASE 6: DISPLAY GAME STATE ---
		printGrid(nrows, ncols, ppSlaveGrid, master_col);
		
		// Show active cannonballs
		int shots = 0;
		for(int i = 0; i < MAX_CANNONBALLS; i++) {
			if(player_cannonballs[i].is_active || invader_cannonballs[i].is_active) shots++;
		}
		if(shots > 0) {
			printf("Cannonballs in flight:\n");
			for(int i = 0; i < MAX_CANNONBALLS; i++) {
				if(player_cannonballs[i].is_active)
					printf("  ↑ Player → (%d,%d) [%ds]\n", player_cannonballs[i].target_row, 
					       player_cannonballs[i].col, player_cannonballs[i].time_to_hit);
				if(invader_cannonballs[i].is_active)
					printf("  ↓ Invader col %d [%ds]\n", invader_cannonballs[i].col, 
					       invader_cannonballs[i].time_to_hit);
			}
		}
		printf("Invaders remaining: %d\n\n", SlaveActiveCount);
		
		// CHECK WIN CONDITION
		if(SlaveActiveCount == 0 && !terminateSignalSent){
			printf("========================================\n");
			printf("  ALL INVADERS DESTROYED! YOU WIN!\n");
			printf("========================================\n");
			terminateSignalSent = 1;
			
			// Send terminate to all slaves
			for(int i = 0; i < (global_size-1); i++){
				send_Val = MSG_TERMINATE;
				MPI_Send(&send_Val, 1, MPI_INT, i, MSG_TERMINATE, world_comm);
			}
			
			// Wait for all slaves to acknowledge termination
			while(SlaveTerminateCount < (global_size-1)) {
				MPI_Iprobe(MPI_ANY_SOURCE, MSG_TERMINATE, world_comm, &flag, &probe_status);
				if(flag) {
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_TERMINATE, world_comm, &status);
					ppSlaveGrid[probe_status.MPI_SOURCE][0] = MSG_TERMINATE;
					SlaveTerminateCount++;
				}
			}
			
			break;  // Exit game loop
		}
		
		
		sleep(1);  // 1 tick = 1 second
	}
	
	printf("Simulation Complete - All processes terminated\n");
	
	// Cleanup
	for(int i = 0; i < (nrows * ncols); i++)
		free(ppSlaveGrid[i]);
	free(ppSlaveGrid);
	
	return 0;
}

// SLAVE PROCESS - Invader Logic
int slave_io(MPI_Comm world_comm, MPI_Comm comm, int nrows, int ncols)
{
	// === INITIALIZATION ===
	int ndims = 2, global_size, global_rank, local_size, local_rank, my_cart_rank;
	int nbr_i_lo, nbr_i_hi, nbr_j_lo, nbr_j_hi;
	MPI_Comm comm2D, col_comm;
	int dims[ndims], coord[ndims], wrap_around[ndims];
	int randNum, avgNum, recvNum_left = 0, recvNum_right = 0, recvNum_top = 0, recvNum_bottom = 0;
	int tick = 0, loop = 1, isDisabled = 0;  // tick will be received via MPI_Bcast
	int my_col_rank, col_size;
	int master_rank = global_size - 1;  // Master is highest rank
	
	MPI_Status status, probe_status;
	MPI_Request send_request[4], receive_request[4];
	int flag = 0, recv_Val;
	
	MPI_Comm_size(world_comm, &global_size);
	MPI_Comm_rank(world_comm, &global_rank);
	MPI_Comm_size(comm, &local_size);
	MPI_Comm_rank(comm, &local_rank);
	
	// Seed random number generator uniquely per process
	srand(time(NULL) + global_rank);
	
	// Set grid dimensions
	if(nrows == 0 && ncols == 0){
		nrows = ncols = (int)sqrt(local_size);
		dims[0] = dims[1] = 0;
	} else {
		dims[0] = nrows;
		dims[1] = ncols;
	}
	
	// Create 2D Cartesian topology
	MPI_Dims_create(local_size, ndims, dims);
	wrap_around[0] = wrap_around[1] = 0;
	MPI_Cart_create(comm, ndims, dims, wrap_around, 0, &comm2D);
	
	// Get coordinates and neighbors
	MPI_Cart_coords(comm2D, local_rank, ndims, coord);
	MPI_Cart_rank(comm2D, coord, &my_cart_rank);
	MPI_Cart_shift(comm2D, SHIFT_ROW, DISP, &nbr_i_lo, &nbr_i_hi);
	MPI_Cart_shift(comm2D, SHIFT_COL, DISP, &nbr_j_lo, &nbr_j_hi);
	
	// Replace MPI_PROC_NULL with NO_NEIGHBOR
	nbr_i_lo = (nbr_i_lo == MPI_PROC_NULL) ? NO_NEIGHBOR : nbr_i_lo;
	nbr_i_hi = (nbr_i_hi == MPI_PROC_NULL) ? NO_NEIGHBOR : nbr_i_hi;
	nbr_j_lo = (nbr_j_lo == MPI_PROC_NULL) ? NO_NEIGHBOR : nbr_j_lo;
	nbr_j_hi = (nbr_j_hi == MPI_PROC_NULL) ? NO_NEIGHBOR : nbr_j_hi;
	
	// === CREATE COLUMN COMMUNICATOR ===
	// This allows invaders in same column to coordinate (bottom-most detection)
	int remain_dims[2] = {1, 0};  // Keep rows (1), collapse columns (0)
	MPI_Cart_sub(comm2D, remain_dims, &col_comm);
	MPI_Comm_rank(col_comm, &my_col_rank);
	MPI_Comm_size(col_comm, &col_size);

	// === MAIN INVADER LOOP ===
	while(loop) {
		// Receive synchronized tick from master (blocks until master broadcasts)
		MPI_Bcast(&tick, 1, MPI_INT, master_rank, MPI_COMM_WORLD);
		
		// Check for messages from master
		MPI_Iprobe((global_size-1), MPI_ANY_TAG, world_comm, &flag, &probe_status);
		if(flag){
			flag = 0;
			switch(probe_status.MPI_TAG)
			{
				case MSG_DISABLE:
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_DISABLE, world_comm, &status);
					recv_Val = MSG_DISABLE;
					MPI_Send(&recv_Val, 1, MPI_INT, global_size-1, MSG_DISABLE, world_comm);
					isDisabled = 1;
					break;
					
				case MSG_TERMINATE:
					MPI_Recv(&recv_Val, 1, MPI_INT, probe_status.MPI_SOURCE, MSG_TERMINATE, world_comm, &status);
					recv_Val = MSG_TERMINATE;
					MPI_Send(&recv_Val, 1, MPI_INT, global_size-1, MSG_TERMINATE, world_comm);
					loop = 0;  // Exit after acknowledging
					break;
					
				default:
					printf("Slave %d: Invalid Tag\n", global_rank);
					break;
			}
		}

		if(!loop) break;  // Exit immediately if terminated
		
		// Exchange random numbers with neighbors (for demonstration)
		randNum = rand() % 50;
		
		MPI_Isend(&randNum, 1, MPI_INT, nbr_i_lo, 0, comm2D, &send_request[0]);
		MPI_Isend(&randNum, 1, MPI_INT, nbr_i_hi, 0, comm2D, &send_request[1]);
		MPI_Isend(&randNum, 1, MPI_INT, nbr_j_lo, 0, comm2D, &send_request[2]);
		MPI_Isend(&randNum, 1, MPI_INT, nbr_j_hi, 0, comm2D, &send_request[3]);

		MPI_Irecv(&recvNum_top, 1, MPI_INT, nbr_i_lo, 0, comm2D, &receive_request[0]);
		MPI_Irecv(&recvNum_bottom, 1, MPI_INT, nbr_i_hi, 0, comm2D, &receive_request[1]);
		MPI_Irecv(&recvNum_left, 1, MPI_INT, nbr_j_lo, 0, comm2D, &receive_request[2]);
		MPI_Irecv(&recvNum_right, 1, MPI_INT, nbr_j_hi, 0, comm2D, &receive_request[3]);
		
		sleep(1);
		
		// Test and cancel incomplete requests
		int test_flag;
		for(int i = 0; i < 4; i++){
			MPI_Test(&send_request[i], &test_flag, MPI_STATUS_IGNORE);
			if(!test_flag) MPI_Cancel(&send_request[i]);
			MPI_Test(&receive_request[i], &test_flag, MPI_STATUS_IGNORE);
			if(!test_flag) MPI_Cancel(&receive_request[i]);
		}

		// If alive, send reports and potentially fire (every 4 ticks)
		if(!isDisabled && (tick % 4 == 0)){
			avgNum = (recvNum_left + recvNum_right + recvNum_top + recvNum_bottom) / 4;
			sleep(2 + coord[0]);
			MPI_Send(&avgNum, 1, MPI_INT, global_size-1, MSG_REPORT, world_comm);
			
			// === DECENTRALIZED FIRING LOGIC ===
			// Determine if I'm bottom-most in my column using column communicator
			// Bottom-most has highest row coordinate in column
			int am_bottom_most = (coord[0] == nrows - 1);
			
			// Could also use MPI_Allreduce on col_comm to find max row, but simple check works for now
			if(am_bottom_most && (rand() % 100) < 10) {  // 10% chance
				int travel_time = 2 + (nrows - 1 - coord[0]);
				int msg_data[2] = {travel_time, coord[1]};
				MPI_Send(msg_data, 2, MPI_INT, global_size-1, MSG_CANNONBALL_FROM_INVADER, world_comm);
		}
	}
	// No sleep needed - MPI_Bcast synchronizes with master's tick
}

// Cleanup
	MPI_Comm_free(&col_comm);
	MPI_Comm_free(&comm2D);
	return 0;
}
