#include <iostream>
#include <cstdlib>
#include <string>

using namespace std;

int GAP = -2, MATCH = 2, MISSMATCH=-1;
int WIDTH, HEIGHT;

int *M;
#define M(i, j) (M[(i) * WIDTH + (j)])


int printMatrix(string seq1, string seq2)
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

int scoreMatrix(string seq1, string seq2)
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

int initializeMatrix(string seq1, string seq2)
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

int traceback(string seq1, string seq2)
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


    cout << align << endl;
    cout << ref << endl;

}


int main(int argc, char *argv[])
{

    string seq1("GAATTCAGTTA"); //First Sequence
    string seq2("GGATCGA"); //Second Sequence

    HEIGHT = seq1.length() + 1;
    WIDTH = seq2.length() + 1;
    M = (int*) malloc(sizeof(int) * (WIDTH) * (HEIGHT));

    initializeMatrix(seq1, seq2);

    scoreMatrix(seq1, seq2);

    printMatrix(seq1, seq2);

    traceback(seq1, seq2);
   
    free(M);


    return 0;
}
