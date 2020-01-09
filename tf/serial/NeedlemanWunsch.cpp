#include <iostream>
#include <string>
using namespace std;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

int M[100][100];
int GAP;
int MATCH;
int MISSMATCH;

int printMatrix(string seq1, string seq2)
{
    int seq1len = seq1.length();
    int seq2len = seq2.length();

    for(int i = 0; i < seq1len + 1; ++i)
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
        
        for(int j = 0; j < seq2len + 1 ; ++j)
        {

            if ((M[i][j] >= 0) || (M[i][j] >= 10))
            {
                printf(" %d\t", M[i][j]);
                continue;
            }

            printf("%d\t", M[i][j]);

        }
        printf("\n");
    }
}

int scoreMatrix(string seq1, string seq2)
{
    int seq1len = seq1.length();
    int seq2len = seq2.length();

    for (int i = 1; i < seq1len + 1; i++)
    {
        for (int j = 1; j < seq2len + 1; j++)
        {
            int scoreDiag = 0;

            if (seq1[i - 1] == seq2[j - 1]){
                scoreDiag = M[i - 1][j - 1] + MATCH;
            }
            else{
                scoreDiag = M[i - 1][j - 1] + MISSMATCH;
            }

            int scoreLeft = M[i][j - 1] + GAP;
            int scoreUp =  M[i - 1][j] + GAP;

            int maxScore = MAX(MAX(scoreDiag, scoreLeft), scoreUp);
            M[i][j] = maxScore;
        }
    }
}

int initializeMatrix(string seq1, string seq2)
{
    int seq1len = seq1.length();
    int seq2len = seq2.length();

    for(int i = 0; i < seq1len + 1; i++)
    {

        M[i][0] = i == 0 ? 0 : i * (GAP);
    }

    for(int j = 0; j < seq2len + 1; j++)
    {

        M[0][j] = j == 0 ? 0 : j * (GAP);
    }

}

int traceback(string seq1, string seq2)
{
    int i = seq1.length();
    int j = seq2.length();

    string align;
    string ref;
    string v, w;

    int scoreDiag;

    // while(i > 0 && j > 0)
    // {
    //     v = &seq1[i-1];
    //     w = &seq2[j-1];
    //
    //     if (seq1[i-1] == seq2[j-1])
    //         scoreDiag = MATCH;
    //     else
    //         scoreDiag = MISSMATCH;
    //
    //     if (i > 0 && j > 0 && M[i][j] == M[i-1][j-1] + scoreDiag)
    //     {
    //         strcpy(&align[alignidx], v);
    //         strcpy(&ref[refidx], w);
    //
    //         i--;
    //         j--;
    //         alignidx--;
    //         refidx--;
    //     }
    //     else if (i > 0 && M[i][j] == M[i-1][j] + GAP)
    //     {
    //         strcpy(&align[alignidx], v);
    //         strcpy(&ref[refidx], "-");
    //
    //         i--;
    //         alignidx--;
    //         refidx--;
    //     }
    //     else if (j > 0 && M[i][j] == M[i][j-1] + GAP)
    //     {
    //         strcpy(&align[alignidx], "-");
    //         strcpy(&ref[refidx], w);
    //
    //         j--;
    //         alignidx--;
    //         refidx--;
    //     }

    // }

    /** printf("%c\n", seq1[i-1]); */
    /** printf("%c\n---", seq2[j-1]); */

    // printf("%s\n", align);
    // printf("%s\n", ref);

}


int main(int argc, char *argv[])
{

    string seq1("GAATTCAGTTA"); //First Sequence
    string seq2("GGATCGA"); //Second Sequence

    GAP = -2;
    MATCH = 2;
    MISSMATCH = -1;

    cout << seq1 << endl;

    initializeMatrix(seq1, seq2);

    scoreMatrix(seq1, seq2);

    printMatrix(seq1, seq2);

    traceback(seq1, seq2);


    return 0;
}
