// v0.2 modified by WZ

//#include <wb.h>
#include "wb4.h" // use our lib instead (under construction)

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

//@@ INSERT CODE HERE

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h) 
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) 
  {
      int pixValR = 0;
      int pixValG = 0;
      int pixValB = 0;
      int pixels = 0;

      for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
      {
          for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
          {
              int curRow = y + blurRow;
              int curCol = x + blurCol;

              if((curRow > -1) && (curRow < h) && (curCol > -1) && (curCol < w))
              {
                  pixValR+= in[(curRow * w + curCol) * 3];
                  pixValG+= in[(curRow * w + curCol) * 3 + 1];
                  pixValB+= in[(curRow * w + curCol) * 3 + 2];
                  pixels++;
              }
          }
      }
      int idxR = (y * w + x) * 3;
      int idxG = (y * w + x) * 3 + 1;
      int idxB = (y * w + x) * 3 + 2;
      out[idxR] = (unsigned char)(pixValR/pixels);
      out[idxG] = (unsigned char)(pixValG/pixels);
      out[idxB] = (unsigned char)(pixValB/pixels);

  }
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

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 1);
  printf( "imagem de entrada: %s\n", inputImageFile );

//  inputImage = wbImportImage(inputImageFile);
  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

// NOW: input and output images are RGB (3 channel)
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
 cudaMemcpy(deviceInputImageData, hostInputImageData,
            imageWidth * imageHeight * sizeof(unsigned char) * 3,
            cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  
  int blockSize = 32;
  dim3 dimGrid((imageWidth-1)/blockSize + 1, (imageHeight-1)/blockSize+1, 1);
  dim3 dimBlock(blockSize, blockSize, 1);

  blurKernel<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, 
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
  /* wbExport( "blurred.ppm", outputImage ); */

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
