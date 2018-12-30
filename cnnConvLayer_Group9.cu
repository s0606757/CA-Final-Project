// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"
using namespace std;

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int sum, ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
	int outArea = FMSIZE/2 * FMSIZE/2;

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 2x2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++){
					for(x = 0; x < 2; x++){
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
short *devInputA, *devInputB;
int *convOut;
int *devOut;

void initGPU()
{   	
	cudaMalloc(&devInputA, sizeof(short)* FMDEPTH * FMSIZE * FMSIZE );
	cudaMalloc(&devInputB, sizeof(short)*FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM );
	cudaMalloc(&convOut, sizeof(int)* FMSIZE * FMSIZE *FILTNUM  );
	cudaMalloc(&devOut, sizeof(int)* FMSIZE * FMSIZE *FILTNUM /4  );

	cudaMemcpy(devInputA, inNeu, sizeof(short)* FMDEPTH * FMSIZE * FMSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devInputB, filt, sizeof(short)*FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM, cudaMemcpyHostToDevice);
	
}

__global__
void convLayerGPU(short *inNeu , short *filt , int *outNeu)
{
    int fn, sli, fmy, fmx, y, x;
	int sum, ifmy, ifmx;
	int filtIdx, inNeuIdx, outNeuIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
   
    fmx=threadIdx.x;
    fmy=threadIdx.y;
    fn=blockIdx.x;
	// Convolution
	sum = 0;
	for(sli = 0; sli < FMDEPTH; sli++){
		for(y = 0; y < FILTSIZE; y++){
			for(x = 0; x < FILTSIZE; x++){
				ifmx = fmx - FILTSIZE / 2 + x;
				ifmy = fmy - FILTSIZE / 2 + y;
				filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
				inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
				if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
					sum += filt[filtIdx] * inNeu[inNeuIdx];
			}
		}
	}
	// Activation - ReLU
	outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
	if(sum <= 0)
		outNeu[outNeuIdx] = 0;
	else
		outNeu[outNeuIdx] = sum;
				
}

__global__
void poolLayerGPU(int *inNeu , int *outNeu)
{	
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;
    int block = blockIdx.x;
	int inNeu_ID = block*32*32 + Ty*32*2 + Tx*2 ;
	int Max = inNeu[inNeu_ID];

	if( Max < inNeu[inNeu_ID+1] )
		Max = inNeu[inNeu_ID+1];
	else
		Max = Max;
	 __syncthreads();
	if( Max < inNeu[inNeu_ID+32] )
		Max = inNeu[inNeu_ID+32];
	else
		Max = Max;
	 __syncthreads();
	if( Max < inNeu[inNeu_ID+33] )
		Max = inNeu[inNeu_ID+33];
	else
		Max = Max;

	 __syncthreads();
	outNeu[block*16*16 + Ty*16 + Tx] = Max;
	
}
/***	Implement your CUDA Kernel here	***/

int main()
{
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	

	timespec time_begin, time_end;                                                 
  clock_gettime(CLOCK_REALTIME, &time_begin);

	convLayerCPU();

  clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = " 
			 <<  convLayerCPUExecTime / 1000 << "ms" << endl;

  
	/***	Lunch your CUDA Kernel here	***/
	
	initGPU();
	clock_gettime(CLOCK_REALTIME, &time_begin);
	
	convLayerGPU<<<512,dim3(32,32)>>>(devInputA,devInputB,convOut); // Launch the kernel
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	
	dim3 threadsPerBlock_pool(FMSIZE/2,FMSIZE/2);
	dim3 numBlocks_pool(FMDEPTH,1);
	poolLayerGPU<<<numBlocks_pool,threadsPerBlock_pool>>>(convOut,devOut);
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	
	cudaMemcpy(outGPU, devOut , sizeof(int) * FMSIZE * FMSIZE *FILTNUM  /4, cudaMemcpyDeviceToHost);
	
	cudaFree(&devInputA);
	cudaFree(&devInputB);
	cudaFree(&devOut);
	
	
	
	/***	Lunch your CUDA Kernel here	***/
  clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " 
			 << convLayerGPUExecTime / 1000 << "ms" << endl;

	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	ending();
	
	return 0;
}
