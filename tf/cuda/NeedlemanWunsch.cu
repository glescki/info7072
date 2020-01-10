#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <string>

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

__global__ void needlemanKernel(int *M, char *seq1, char *seq2, int width, 
                                int height, int MATCH, int MISMATCH, int GAP) 
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0 | j == 0 | i >= height | j >= width)
		return;

    int scoreDiag = 0;

	if (seq1[i - 1] == seq2[j - 1]) 
		scoreDiag = M[(i - 1) * width + (j - 1)] + MATCH;
	else
		scoreDiag = M[(i - 1) * width + (j - 1)] + MISMATCH;
	
	int scoreLeft = M[i * width + (j - 1)] + GAP;
	int scoreUp = M[(i - 1) * width + j] + GAP;

	M[i * width + j] = max(max(scoreDiag, scoreLeft), scoreUp);
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
    
    cudaMalloc((void *)&deviceM, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&deviceSeq1, seq1.length());
    cudaMalloc(&deviceSeq2, seq2.length());

    cudaMemcpy(deviceM, M, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSeq1, hostSeq1, WIDTH * HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSeq2, hostSeq2, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    int blockSize = 32;                                                           
    dim3 dimGrid((WIDTH-1)/blockSize + 1, (HEIGHT-1)/blockSize+1, 1);   
    dim3 dimBlock(blockSize, blockSize, 1);                                       
                                                                                
    needlemanKernel<<<dimGrid,dimBlock>>>(deviceM, deviceSeq1, deviceSeq2, WIDTH,
                                          HEIGHT, MATCH, MISMATCH, GAP);   

    cudaMemcpy(M, deviceM, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceM);
    cudaFree(deviceSeq1);
    cudaFree(deviceSeq2);

    // traceback(seq1, seq2);

    free(M);

    return 0;
}
