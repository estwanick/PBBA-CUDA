/* Parallel Best Band Selection Algorithm */
/*
Max Bands searched: 45 -> Taking Approximately a Day or so to complete 
Value returned: 0.094479 
Band Returning Max: UNKNOWN 

author: Michael C Estwanick 
*/

#include <thrust/extrema.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/timeb.h>


#define ARRAYSIZE 45 //Number of Bands to check 
#define TOTAL powf(2,ARRAYSIZE)

//Cuda Error Check
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Kernel that Performs the best band selection algorithm 
__global__ void kernel(float *cc, long long int jump, float threadCount){

    
    int N = threadCount*threadCount; // Total threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int threadId = col + row * N; // Two dimensional Thread Index 

    int decimalNumber,quotient;
    int binaryNumber[ARRAYSIZE]; //Holds the binary number in an array

    //Spectra Data Set
    int a[169] = {1192, 1315, 1462, 1484, 1476, 1443, 1508, 1489, 1470, 1537, 1633, 1539, 1600, 1707, 1701, 1682, 1688, 1681, 1694, 1728, 1786, 1821, 1830, 1881, 1893, 1816, 1692, 1675, 1651, 1579, 1514, 1600, 1576, 1543, 1465, 1440, 1452, 1483, 1944, 2303, 2616, 3118, 3861, 4054, 3915, 4790, 5543, 4539, 4679, 5574, 5365, 5080, 4186, 4272, 4934, 5057, 5000, 4867, 3872, 2992, 2519, 1203, 1092, 1979, 3005, 3886, 4121, 4134, 4168, 4014, 3612, 3391, 2712, 1324, 473, 556, 1099, 1769, 1979, 2063, 2289, 2494, 2553, 2196, 2125, 2147, 1749, 1221, 667, 517, 732, 885, 988, 1051, 1001, 984, 997, 965, 1008, 1022, 992, 993, 982, 946, 850, 698, 562, 446, 334, 278, 226, 161, 99, 58, 125, 139, 101, 93, 115, 151, 167, 171, 178, 172, 180, 176, 163, 152, 143, 134, 129, 130, 139, 148, 151, 146, 137, 123, 106, 98, 79, 65, 58, 70, 60, 62, 51};

int b[169] = {1162, 1337, 1282, 1491, 1508, 1517, 1488, 1513, 1539, 1576, 1626, 1634, 1573, 1786, 1741, 1782, 1755, 1669, 1700, 1826, 1832, 1895, 1920, 1938, 1933, 1852, 1808, 1806, 1747, 1718, 1628, 1659, 1639, 1621, 1589, 1525, 1526, 1583, 2118, 2549, 2900, 3411, 4237, 4340, 4126, 4985, 5760, 4716, 4840, 5793, 5616, 5326, 4416, 4485, 5197, 5322, 5315, 5166, 4107, 3158, 2664, 1286, 1149, 2093, 3197, 4157, 4413, 4422, 4444, 4287, 3842, 3620, 2892, 1415, 498, 591, 1164, 1892, 2110, 2215, 2441, 2663, 2721, 2351, 2286, 2296, 1872, 1318, 714, 568, 805, 977, 1084, 1143, 1094, 1071, 1085, 1044, 1092, 1116, 1070, 1076, 1068, 1031, 928, 766, 617, 481, 370, 305, 250, 181, 108, 64, 139, 153, 109, 101, 122, 162, 180, 189, 192, 191, 195, 192, 178, 164, 153, 145, 141, 139, 148, 158, 163, 151, 148, 131, 120, 107, 91, 71, 72, 81, 65, 66, 62};

    //Holds Product from the dot product
    int c[ARRAYSIZE];
    //Arrays to hold integers summed 
    int aSumArr[ARRAYSIZE];
    int bSumArr[ARRAYSIZE];

    //Initialize arrays 
    for(int i = 0; i < ARRAYSIZE; i++){
        c[i] = 0;
        aSumArr[i] = 0;
        bSumArr[i] = 0;
        binaryNumber[i] = 0;
    }
                                                                                                                                                                                                                             
    
    int dotSum = 0; //value for the dot product
    int aSum = 0; //sum of valid array positions for array a
    int bSum = 0; //sum of valid array positions for array b
    int i = 0;
    float finalValue = 0; //Value of the arcCos of the dot product / sqrt(array a) * sqrt(array b)

    //Add jump to decimal to avoid running combinations that have already been calculated 
    decimalNumber = threadId + jump;
    quotient = decimalNumber;

    //Loop to convert decimal into binary and store in array
    while(quotient!=0){
        binaryNumber[i++]= quotient % 2;
        quotient = quotient / 2;
    }

    //Loop through binaryNumber array
    for(int x = ARRAYSIZE-1 ; x >= 0; x--){
        //Only perform calculation on selected bands
        if(binaryNumber[x] == 1){
            //Perform multiplication for dot product
            c[x] = a[x] * b[x];
            //Fill sum arrays at correct index
            aSumArr[x] = a[x];
            bSumArr[x] = b[x];
        }else{
            //Do Nothing
        }
    }

    //Sums up the product array to complete dot product
    for(int j = 0; j < ARRAYSIZE; ++j){
        dotSum += c[j]; // Dot Product 
        aSum += powf( aSumArr[j], 2 ); // Euclidean Norm on vector A
        bSum += powf( bSumArr[j], 2 ); // Euclidean Norm on vector B
    }

    //Create values for algorithm 
    float sqSum1 = sqrtf(aSum); //Finish Euclidean Norm on vector A
    float sqSum2 = sqrtf(bSum); //Finish Euclidean Norm on vector B
    float sqSum = sqSum1 * sqSum2; 
    float div = dotSum / sqSum ;
    //Plug in values for final answer
    finalValue = acosf( div ) ;

    //Stores the threads final value in array cc, in the respected index
    if(finalValue == finalValue){ //Check if the result is a real number 
        cc[threadId] = finalValue; //store value in array to be passed back to host (CPU)
    }else{
        cc[threadId] = -2; //If the value return is NaN set result = -2
    }
                                                                                                                                                                                                                             
}//End kernel 

float getFreeMem();
void deviceProperties();
float kernelCount(float freeMem, float totalMem);

int main( void ) {

    printf("------------------------------------------------------ \n");
    printf("2 ^ %d bands \n", ARRAYSIZE);
    cudaDeviceReset();

    float freeMem = getFreeMem(); // Get available free memory
    float kernels = kernelCount( freeMem, TOTAL); // get number of kernels to launch 
    //Number of elements for each kernel
    float threadCount = ( TOTAL / kernels );
    printf("threadCount: Total thread Count: %lf \n", threadCount);
    //number of threads per kernel
    float threadsPerDim =  ceil( powf(threadCount,(.25f)) );
    printf("threadPerDim: Total threads per dimension: %lf \n", threadsPerDim);
    long long int jump = 0;
    
    float *h_c = (float *)malloc(sizeof(float)*threadCount); //Host Vector
    float *d_c; //Device Vector 
    //Collection of individual kernel max 
    float *maxCollection = (float *)malloc(sizeof(float)*kernels);

    float totalTime = 0.0;

    //CPU Timer start
    struct timeb start, end;
    int diff;
    ftime(&start);

    //Loop Through the kernel as many times needed to execute all bands,
    //When The GPU is out of memory the loop will execute again storing 
    //The max from each subset of bands in the maxCollection array
    for(int i = 0; i < kernels; i++){
        cudaDeviceReset();
  
        //Setup Thread & Block Grid 
        dim3 blocks (threadsPerDim, threadsPerDim);
        dim3 threads (threadsPerDim, threadsPerDim);

        //Allocate Device Memory
        HANDLE_ERROR( cudaMalloc((void**)&d_c, sizeof(float)*(threadCount)) );

        //Timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        //Execute Kernel
        kernel<<<blocks, threads>>>(d_c, jump, threadsPerDim);

        //Timer stuff
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        //printf("GPU: Kernel: %d Time:  %f \n", i ,milliseconds/1000);
        totalTime += milliseconds;

        //Retrieve vector from device holding the max value from each subset of bands
        HANDLE_ERROR( cudaMemcpy(h_c, d_c, sizeof(float)*(threadCount), cudaMemcpyDeviceToHost) );

        //Get the max value from the current subset of bands executed using THRUST Library 
        float *result = thrust::max_element(h_c, h_c + (int)threadCount);
        
        //Store the max in the maxCollection array
        maxCollection[i] = *result;
        //printf(" \t Jump Size: %ld \n", jump);
        jump = jump + threadCount; //Increment jump to avoid checking completed bands
        HANDLE_ERROR( cudaFree(d_c) ); 

    }

    //Get max of all kernels 
    float *result = thrust::max_element(maxCollection, maxCollection + (int)kernels);
    //Print the maximum of all bands executed from all the kernels combined 
    printf("Total Max: is: %f \n", *result); 

    //Stop timer
    ftime(&end);
    diff = (int) (1000.0 * (end.time - start.time)
        + (end.millitm - start.millitm));

    printf("\nOperation took %u milliseconds\n", diff);
    //printf("Total GPU Time: %f \n", totalTime/1000 ); 

    return 0;
}
//Return the number of kernels 
float kernelCount(float freeMem, float totalMem){

    float totalSize = sizeof(float) * totalMem ;
    float kernels =  ceil( totalSize / freeMem ) ;

    printf("Total array size %lf || free mem %lf \n", totalSize, freeMem);
    printf("Kernels: %lf \n ", kernels);
    return kernels;
}
//Get available free memory 
float getFreeMem(){

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    //fprintf(stderr, "Free = %ld, Total = %ld\n", freeMem, totalMem);

    return freeMem;
}
//Get total memory of device 
void deviceProperties(){
    cudaDeviceProp  prop;
    int devCount;
    HANDLE_ERROR( cudaGetDeviceCount( &devCount ) );
    for (int i=0; i< devCount; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
    }

        //Number of threads
}
                                                                                                                                                                                                                             
