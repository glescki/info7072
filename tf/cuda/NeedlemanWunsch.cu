#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <chrono>

using namespace std;

int GAP = -1, MATCH = 1, MISMATCH=-1;
int WIDTH, HEIGHT;

int *M;
#define M(i, j) (M[(i) * WIDTH + (j)])


void printMatrix(string seq1, string seq2)
{
    int seq1len = HEIGHT;
    int seq2len = WIDTH;

    for(int i = 0; i < seq1len; ++i)
    {
        if ( i == 0 )
        {
            printf("\t\t");
            for (int j = 0; j < seq2len; ++j)
            {
                printf("%c\t", seq2[j]);
            }
            printf("\n");
        }

        if (i == 0)
            printf("\t");

        if ( i >= 1 )
            printf("%c\t", seq1[i-1]);
        
        for(int j = 0; j < seq2len; ++j)
        {

            if ((M(i, j) >= 0) || (M(i, j) >= 10))
            {
                printf(" %d\t", M(i, j));
                continue;
            }

            printf("%d\t", M(i, j));

        }
        printf("\n");
    }
}


void initializeMatrix()
{
    int seq1len = HEIGHT;
    int seq2len = WIDTH;

    for(int i = 0; i < seq1len; ++i)
    {
        M(i, 0) = i == 0 ? 0 : i * (GAP);
    }

    for(int j = 0; j < seq2len; ++j)
    {

        M(0, j) = j == 0 ? 0 : j * (GAP);
    }

}

void traceback(string seq1, string seq2)
{
    int i = seq1.length();
    int j = seq2.length();

    string align = "";
    string ref = "";
    string v, w;

    int scoreDiag;

    while(i > 0 && j > 0)
    {
        v = seq1[i-1];
        w = seq2[j-1];

        if (seq1[i-1] == seq2[j-1])
            scoreDiag = MATCH;
        else
            scoreDiag = MISMATCH;

        if (i > 0 && j > 0 && M(i, j) == M(i-1, j-1) + scoreDiag)
        {
            align = v + align;
            ref = w + ref;

            i--;
            j--;
        }
        else if (i > 0 && M(i, j) == M(i-1, j) + GAP)
        {
            align = v + align;
            ref = "-" + ref;

            i--;
        }
        else if (j > 0 && M(i, j) == M(i, j-1) + GAP)
        {
            align = "-" + align;
            ref = w + ref;

            j--;
        }

    }


    cout << align << endl;
    cout << ref << endl;

}

__global__ void needlemanKernel(int *devM, char *seq1, char *seq2, int width, 
                                int height, int MATCH, int MISMATCH, int GAP, int k) 
{

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int bdx = blockDim.x;

    int i = by * bdx + tx;
    int j = -i + width - (width - 2) + k;


    if (i >= height || i <= 0 || j >= width || j <= 0)
        return;


    int scoreDiag = 0;

    if (seq1[i - 1] == seq2[j - 1])
        scoreDiag = devM[(i - 1) * width + (j - 1)] + MATCH;
    else
    {
        // printf("%d, %d\n", i, j);
        //
        // printf("%c %c", seq1[i-1], seq2[j-1]);
        // printf("\n");
        scoreDiag = devM[(i - 1) * width + (j - 1)] + MISMATCH;

    }

    int scoreLeft = devM[i * width + (j - 1)] + GAP;
    int scoreUp = devM[(i - 1) * width + j] + GAP;

    devM[i * width + j] = max(max(scoreDiag, scoreLeft), scoreUp);

    __syncthreads();
}


int main(int argc, char *argv[])
{
    int c;
    string seq1;    
    string seq2;

    if(argc < 1)
    {
        seq1 = "GAATTCAGTTA"; //First Sequence
        seq2 = "GGATCGA"; //Second Sequence
    }

    while((c = getopt(argc, argv, ":r:")) != -1 ) 
    {
        switch(c)
        {
            case 'r':
                seq1 = optarg;
                break;
            case '?':
                printf("%c\n", optopt);
                break;
            default:
                break;
        }
    }

    for (int index = optind; index < argc; ++index)
        seq2 = argv[index];
         

    HEIGHT = seq1.length() + 1;
    WIDTH = seq2.length() + 1;
    M = (int*) malloc(sizeof(int) * (WIDTH) * (HEIGHT));

    initializeMatrix();

    char hostSeq1[seq1.length()];
    char hostSeq2[seq2.length()];

    strcpy(hostSeq1, seq1.c_str());
    strcpy(hostSeq2, seq2.c_str());

    int *deviceM;
    char *deviceSeq1;
    char *deviceSeq2;
    
    cudaMalloc(&deviceM, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&deviceSeq1, seq1.length());
    cudaMalloc(&deviceSeq2, seq2.length());

    cudaMemcpy(deviceM, M, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSeq1, hostSeq1, seq1.length(), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSeq2, hostSeq2, seq2.length(), cudaMemcpyHostToDevice);

    int nThreads = 1024;
    
    dim3 dimGrid(1, (HEIGHT-1)/nThreads+1, 1);   
    dim3 dimBlock(HEIGHT, 1, 1);                                       
                                                                                
    auto t1 = chrono::high_resolution_clock::now();

    for(int k = 0; k < (WIDTH + HEIGHT) - 1; ++k)
    {
        needlemanKernel<<<dimGrid,dimBlock>>>(deviceM, deviceSeq1, deviceSeq2,
                                              WIDTH, HEIGHT, MATCH, MISMATCH, GAP,
                                              k);
    }

    auto t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    cout << duration << endl;  

    cudaMemcpy(M, deviceM, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceM);
    cudaFree(deviceSeq1);
    cudaFree(deviceSeq2);

    
    // printMatrix(seq1, seq2);
    // traceback(seq1, seq2);

    free(M);

    return 0;
}
