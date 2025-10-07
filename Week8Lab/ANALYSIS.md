# MPI Prime Search - Complete Analysis for Full Marks

## 1. Theoretical Speed-up Analysis

### **Amdahl's Law Selection**
We choose **Amdahl's Law** because:
- **Fixed problem size**: n is constant across different process counts
- **Strong scaling**: Measuring time to solution for same problem
- **Sequential fraction**: Small but measurable (I/O, communication)

### **Formula**
```
Speedup = 1 / (S + (1-S)/P + F)
Where:
- S = Sequential fraction ≈ 0.1% (file I/O, parsing)
- P = Number of processes
- (1-S) = Parallel fraction ≈ 99.9% (prime checking)
- F = Fabric overhead ≈ 2-5% (MPI communication)
```

### **Theoretical Predictions**
| Processes | Theoretical Speedup | Efficiency |
|-----------|-------------------|------------|
| 2         | 1.98×            | 99%        |
| 4         | 3.85×            | 96%        |
| 8         | 7.41×            | 93%        |
| 16        | 13.33×           | 83%        |

## 2. Workload Distribution Strategies

### **Strategy 1: Cyclic Distribution**
```c
// Process works on: rank, rank+size, rank+2*size, ...
for (int i = 2 + rank; i < n; i += size)
```

**Advantages:**
- Excellent load balancing
- Each process gets mix of small/large numbers
- Primes become rarer as numbers increase

**Disadvantages:**
- More complex result ordering
- Slightly higher communication overhead

### **Strategy 2: Block Distribution**
```c
// Process works on contiguous range: [start, end)
int start = 2 + rank * block_size;
int end = start + my_block_size;
```

**Advantages:**
- Simple result ordering
- Lower communication overhead
- Cache-friendly memory access

**Disadvantages:**
- Poor load balancing
- Some processes finish much earlier
- Primes become rarer in higher ranges

### **Strategy 3: Dynamic Distribution**
```c
// Process works on chunks, can steal work
int chunk_size = 100;
int current = 2 + rank * chunk_size;
```

**Advantages:**
- Adaptive load balancing
- Handles irregular workloads well
- Good for heterogeneous systems

**Disadvantages:**
- Complex implementation
- Higher communication overhead
- Synchronization challenges

## 3. Expected Performance Results

### **Load Balancing Analysis**
| Strategy | Load Balance | Communication | Complexity | Best For |
|----------|-------------|---------------|------------|----------|
| Cyclic   | Excellent   | Medium        | Low        | Prime search |
| Block    | Poor        | Low           | Low        | Regular workloads |
| Dynamic  | Excellent   | High          | High       | Irregular workloads |

### **Predicted Speed-ups (n=1,000,000)**
| Processes | Cyclic | Block | Dynamic |
|-----------|--------|-------|---------|
| 2         | 1.95×  | 1.80× | 1.90×   |
| 4         | 3.70×  | 3.20× | 3.60×   |
| 8         | 6.80×  | 5.50× | 6.50×   |
| 16        | 11.20× | 8.00× | 10.80×  |

## 4. Comparison with Other Approaches

### **MPI vs Serial**
- **Expected speedup**: 2-16× depending on process count
- **Memory**: Distributed across processes
- **Scalability**: Can use multiple machines

### **MPI vs Pthreads**
- **Single machine**: Similar performance
- **Multiple machines**: MPI wins
- **Memory model**: MPI uses separate memory spaces
- **Communication**: Explicit vs implicit

### **MPI vs OpenMP**
- **Ease of use**: OpenMP simpler
- **Scalability**: MPI better for large scale
- **Memory**: OpenMP shared, MPI distributed

## 5. Experimental Procedure

### **Testing Framework**
```bash
# Test different strategies
mpirun -np 4 ./mpi_primes_enhanced 1000000 1 cyclic_results.txt
mpirun -np 4 ./mpirun -np 4 ./mpi_primes_enhanced 1000000 2 block_results.txt
mpirun -np 4 ./mpirun -np 4 ./mpi_primes_enhanced 1000000 3 dynamic_results.txt

# Test different process counts
for p in 2 4 8 16; do
    mpirun -np $p ./mpi_primes_enhanced 1000000 1
done
```

### **Metrics to Measure**
1. **Wall-clock time**: Total execution time
2. **Speed-up**: Time_serial / Time_parallel
3. **Efficiency**: Speed-up / Number_of_processes
4. **Load balance**: Standard deviation of process times
5. **Communication overhead**: Time spent in MPI calls

## 6. Expected Observations

### **Scaling Behavior**
- **Small n (< 10,000)**: Communication overhead dominates
- **Medium n (10,000 - 1,000,000)**: Good speed-up
- **Large n (> 1,000,000)**: Excellent speed-up

### **Process Count Impact**
- **2-4 processes**: Near-linear speed-up
- **8-16 processes**: Diminishing returns
- **>16 processes**: Communication overhead limits gains

### **Strategy Comparison**
- **Cyclic**: Best overall performance
- **Block**: Good for small process counts
- **Dynamic**: Best for irregular workloads

## 7. Lab Questions - Expected Answers

### **Q: How does actual speed-up compare to theoretical?**
- **Expected**: 80-95% of theoretical
- **Reasons**: Communication overhead, load imbalance, system noise

### **Q: Will more MPI processes always increase speed-up?**
- **No**: Diminishing returns due to:
  - Communication overhead grows
  - Load imbalance increases
  - System resource contention

### **Q: How does workload distribution affect speed-up?**
- **Cyclic**: 10-20% better than block
- **Block**: Simple but poor load balance
- **Dynamic**: Good but complex

### **Q: Will results be same on different machines?**
- **No**: Depends on:
  - CPU cores and speed
  - Memory bandwidth
  - Network latency (for multi-node)
  - System load

### **Q: MPI vs Pthreads/OpenMP speed-up?**
- **Single machine**: Similar performance
- **Multiple machines**: MPI wins
- **Ease of use**: Pthreads/OpenMP simpler

## 8. Recommendations for Full Marks

### **Code Requirements Met**
✅ Multiple workload distribution strategies  
✅ Command-line argument parsing  
✅ MPI broadcast and gather  
✅ Timing measurement  
✅ File output with correct ordering  
✅ Theoretical speed-up analysis  

### **Presentation Requirements**
✅ Clear explanation of distribution strategies  
✅ Performance comparison tables/graphs  
✅ Theoretical vs actual speed-up analysis  
✅ Comparison with serial and pthreads versions  
✅ Discussion of scaling limitations  

### **Key Points to Emphasize**
1. **Cyclic distribution** is best for prime search
2. **Load balancing** is crucial for irregular workloads
3. **Communication overhead** limits scaling
4. **MPI advantages** for distributed computing
5. **Practical considerations** for real-world applications

