#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <ctime>
#include <chrono>

using namespace std;

int GAP = -1, MATCH = 1, MISSMATCH=-1;
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

void scoreMatrix(string seq1, string seq2)
{
    int seq1len = HEIGHT;
    int seq2len = WIDTH;

    for (int i = 1; i < seq1len; i++)
    {
        for (int j = 1; j < seq2len; j++)
        {
            int scoreDiag = 0;

            if (seq1[i - 1] == seq2[j - 1]){
                scoreDiag = M(i - 1, j - 1) + MATCH;
            }
            else{
                scoreDiag = M(i - 1, j - 1) + MISSMATCH;
            }

            int scoreLeft = M(i, j - 1) + GAP;
            int scoreUp =  M(i - 1, j) + GAP;

            int maxScore = max(max(scoreDiag, scoreLeft), scoreUp);
            M(i, j) = maxScore;
        }
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

    string align = " ";
    string ref = " ";
    string v, w;

    int scoreDiag;

    while(i > 0 && j > 0)
    {
        v = seq1[i-1];
        w = seq2[j-1];

        if (seq1[i-1] == seq2[j-1])
            scoreDiag = MATCH;
        else
            scoreDiag = MISSMATCH;

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

    // cout << align << endl;
    // cout << ref << endl;

}


string genseq(int size)
{
    string sequence = "";
    int r;

    srand(time(NULL));

    for(int i=0; i<size; ++i)
    {
        r = rand() % 10;

        if(r < 3)
        {
            sequence = sequence + "A";
            continue;
        }

        if(r < 6)
        {
            sequence = sequence + "T";
            continue;
        }

        if(r < 8)
        {
            sequence = sequence + "C";
            continue;
        }

        if(r < 10)
        {
            sequence = sequence + "G";
            continue;
        }
    }

    return sequence;
}

int main(int argc, char *argv[])
{

    if(argc < 1)
    {
        string seq1("GAATTCAGTTA"); //First Sequence
        string seq2("GGATCGA"); //Second Sequence
    }

    int c;
    string seq1;
    string seq2;

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
    
    if(seq2.length() < 1)
    {
        cout << "Error: align sequence not in the input" << endl;
        return -1;
    }

    HEIGHT = seq1.length() + 1;
    WIDTH = seq2.length() + 1;
    M = (int*) malloc(sizeof(int) * (WIDTH) * (HEIGHT));

    initializeMatrix();

    auto t1 = chrono::high_resolution_clock::now();
    scoreMatrix(seq1, seq2);
    auto t2 = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    
    cout << duration << endl;

    // traceback(seq1, seq2);

    free(M);


    return 0;
}
