// v0.2 modified by WZ

//#include <wb.h>
#include "wb4.h" // use our lib instead (under construction)
#define GTX480   480
#define GTX680   680
#define GTX750Ti 7502
#define GPUmodel GTX750Ti

#if GPUmodel == GTX480
    #define MP 15   // number of mutiprocessors (SMs) in GTX480
    #define GRID1(MP*2) // GRID sizefor rgb2uintKernelSHM and rgb2uintKernelSHM kernels
    #define NT1 768 // number of threads per block in the 
                    //   rgb2uintKernelSHM and rgb2uintKernelSHM kernels
                    //    this is perhaps the best value for GTX480
#elif GPUmodel == GTX680
    #define MP 8    // number of mutiprocessors (SMs) in GTX680
    #define GRID1(MP*2) // GRID sizefor rgb2uintKernelSHM and rgb2uintKernelSHM kernels
    #define NT1 1024    // number of threads per block in the 
                        //   rgb2uintKernelSHM and rgb2uintKernelSHM kernels
                        //    this is perhaps the best value for GTX680
#elif GPUmodel == GTX750ti
    #define MP 5
    #define GRID1(MP*2)
    #define NT1 1024    

#endif

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE 5
#define NTHx 32
#define NTHy 32
#define TILE_WIDTH 64 - 2*BLUR_SIZE
#define TILE_HEIGHT 64 - 2*BLUR_SIZE


//@@ INSERT CODE HERE
__global__ void rgb2uintKernelSHM(  unsigned int* argb, unsigned char* rgb, 
                                   int w, int h )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x < w && y < h )
  {
      int idx = y * w + x;
      unsigned char r = rgb[idx * 3];
      unsigned char g = rgb[idx * 3 + 1];
      unsigned char b = rgb[idx * 2 + 2];
      unsigned int v = ((unsigned int)r << 16) + ((unsigned int)g << 8) + (unsigned int)b;

      argb[idx] = v;
  }
}

__global__ void uint2rgbKernelSHM(  unsigned int* argb, unsigned char* rgb, 
                                   int w, int h )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x < w && y < h )
  {
      int idx = y * w + x;
      unsigned char r = (unsigned char) (argb[idx] >> 16 & 0xff);
      unsigned char g = (unsigned char) (argb[idx] >> 8 & 0xff);
      unsigned char b = (unsigned char) (argb[idx] & 0xff);

      rgb[idx * 3] = r;
      rgb[idx * 3 + 1] = g;
      rgb[idx * 3 + 2] = b;
  }
}

__global__ void blurKernelSHM(unsigned char* in, unsigned char* out, int w, int h) 
{
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Col = bx * blockDim.x + tx;
  int Row = by * blockDim.y + ty;

  if (Col > h || Row > w)
      return;
   
  const int tile_size = ( TILE_HEIGHT + 2*BLUR_SIZE ) * ( TILE_WIDTH + 2*BLUR_SIZE );
  __shared__ unsigned char tile[ tile_size ];

  int block_size = NTHx * NTHy;

  for ( int i = 0; i < tile_size / block_size; ++i)
  {
    int tile_idx = (ty * blockDim.y + tx) + i * block_size;
    /* tile[tile_idx] =  */
  }

  /* if ( (Row < w && i * TILE_WIDTH + tx < w) || (Col < h && i * TILE_WIDTH + ty < h) ) */
  /* { */
      /* tile[ty][tx] = in[Row * w + ( i * TILE_WIDTH + tx )]; */
  /* } */
  /* else */
  /* { */
      /* tile[ty][tx] = 0.0; */
  /* } */
  /* __syncthreads(); */

  int pixels = 0;
  for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
  {
      for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
      {
          int curRow = Row + blurRow;
          int curCol = Col + blurCol;

          if((curRow > -1) && (curRow < h) && (curCol > -1) && (curCol < w))
          {
              int idx = curRow * w + curCol;
              unsigned char r = (unsigned char) (in[idx] >> 16 & 0xff);
              unsigned char g = (unsigned char) (in[idx] >> 8 & 0xff);
              unsigned char b = (unsigned char) (in[idx] & 0xff);

              pixels++;
          }
      }
  }

  /* int idxR = (y * w + x) * 3; */
  /* int idxG = (y * w + x) * 3 + 1; */
  /* int idxB = (y * w + x) * 3 + 2; */
  /* out[idxR] = (unsigned char)(pixValR/pixels); */
  /* out[idxG] = (unsigned char)(pixValG/pixels); */
  /* out[idxB] = (unsigned char)(pixValB/pixels); */

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  unsigned char *hostInputImageData;
  unsigned char *hostOutputImageData;
  unsigned char *deviceInputImageData;
  unsigned char *deviceOutputImageData;
  unsigned int *deviceInputImageData_argb;
  unsigned int *deviceOutputImageData_argb;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 1);
  printf( "imagem de entrada: %s\n", inputImageFile );

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

// NOW: input and output images are RGB (3 channel)
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  // rgb format image (with pixels as char)
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
 
  // argb format image (with pixels as int)
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned int)); 
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned int));
 
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
 cudaMemcpy(deviceInputImageData, hostInputImageData,
            imageWidth * imageHeight * sizeof(unsigned char) * 3,
            cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  
  dim3 dimGrid((imageWidth-1)/NTHy + 1, (imageHeight-1)/NTHx+1, 1);
  dim3 dimBlock(NTHy, NTHx, 1);


  blurKernelSHM<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData,
                                   imageWidth, imageHeight);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3,
             cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);
  // DEBUG: if you want to see your image, 
  //   will generate file bellow in current directory
  wbExport( "blurred.ppm", outputImage );

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
