//
//   v0.2 corrigida por WZola aug/2017 para ficar de acordo com novo wb.h 
//        (ou seja de acordo com wb4.h)
//        

//#include <wb.h>     // original
// DOWNLOAD wb4.h from the discipline site
#include "./wb4.h" // use our new lib, wherever it is
                                              

#include <string.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE
__global__ void colorConvert(unsigned char* grayImage,unsigned char* rgbImage,
                             int width, int height, int channels)
{
    int x =  threadIdx.x + blockIdx.x * blockDim.x;
    int y =  threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
    {
        int idx = y * width + x;
        unsigned char r =  rgbImage[channels * idx]; // red value for pixel
        unsigned char g =  rgbImage[channels * idx + 1]; // green value for pixel
        unsigned char b =  rgbImage[channels * idx + 2]; // blue value for pixel

        grayImage[idx] = (0.21*r + 0.71*g + 0.07*b);
    }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;

//  float *hostInputImageData;
//  float *hostOutputImageData;
//  float *deviceInputImageData;
//  float *deviceOutputImageData;

  unsigned char *hostInputImageData;
  unsigned char *hostOutputImageData;
  unsigned char *deviceInputImageData;
  unsigned char *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */
//  show_args( args ); // debug

//  inputImageFile = wbArg_getInputFileName(args, 2);
    inputImageFile = argv[2];

//  inputImage = wbImportImage(inputImageFile);
  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels);
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels,
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  int blockSize = 5;
  dim3 dimGrid((imageWidth-1)/blockSize + 1, (imageHeight-1)/blockSize+1, 1);
  dim3 dimBlock(blockSize, blockSize, 1);
  colorConvert<<<dimGrid,dimBlock>>>(deviceOutputImageData, deviceInputImageData, 
                                   imageWidth, imageHeight, imageChannels);

  
  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
