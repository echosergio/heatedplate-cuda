/*
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <cmath>
# include <ctime>
# include <string>
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "calculate_solution_gold.cu"

#define M 50
#define N 50

void initialize_grid(double w[M][N]);

int main(int argc, char *argv[])

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for HEATED_PLATE.
//
//  Discussion:
//
//    This code solves the steady state heat equation on a rectangular region.
//
//    The sequential version of this program needs approximately
//    18/epsilon iterations to complete.
//
//
//    The physical region, and the boundary conditions, are suggested
//    by this diagram;
//
//                   W = 0
//             +------------------+
//             |                  |
//    W = 100  |                  | W = 100
//             |                  |
//             +------------------+
//                   W = 100
//
//    The region is covered with a grid of M by N nodes, and an N by N
//    array W is used to record the temperature.  The correspondence between
//    array indices and locations in the region is suggested by giving the
//    indices of the four corners:
//
//                  I = 0
//          [0][0]-------------[0][N-1]
//             |                  |
//      J = 0  |                  |  J = N-1
//             |                  |
//        [M-1][0]-----------[M-1][N-1]
//                  I = M-1
//
//    The steady state solution to the discrete heat equation satisfies the
//    following condition at an interior grid point:
//
//      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )
//
//    where "Central" is the index of the grid point, "North" is the index
//    of its immediate neighbor to the "north", and so on.
//
//    Given an approximate solution of the steady state heat equation, a
//    "better" solution is given by replacing each interior point by the
//    average of its 4 neighbors - in other words, by using the condition
//    as an ASSIGNMENT statement:
//
//      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )
//
//    If this process is repeated often enough, the difference between successive
//    estimates of the solution will go to zero.
//
//    This program carries out such an iteration, using a tolerance specified by
//    the user, and writes the final estimate of the solution to a file that can
//    be used for graphic processing.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    22 July 2008
//
//  Author:
//
//    Original C version by Michael Quinn.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Michael Quinn,
//    Parallel Programming in C with MPI and OpenMP,
//    McGraw-Hill, 2004,
//    ISBN13: 978-0071232654,
//    LC: QA76.73.C15.Q55.
//
//  Parameters:
//
//    Commandline argument 1, double EPSILON, the error tolerance.
//
//    Commandline argument 2, char *OUTPUT_FILENAME, the name of the file into which
//    the steady state solution is written when the program has completed.
//
//  Local parameters:
//
//    Local, double DIFF, the norm of the change in the solution from one iteration
//    to the next.
//
//    Local, double MEAN, the average of the boundary values, used to initialize
//    the values of the solution in the interior.
//
//    Local, double U[M][N], the solution at the previous iteration.
//
//    Local, double W[M][N], the solution computed at the latest iteration.
//
{
    double diff;
    double epsilon;
    int i;
    int j;
    FILE *output;
    char output_filename[80];
    int success;
    double w[M][N];

    printf("\n");
    printf("HEATED_PLATE <epsilon> <fichero-salida>\n");
    printf("  C/serie version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);

    //
    //  Read EPSILON from the command line or the user.
    //
    epsilon = atof(argv[1]);
    printf("The iteration will be repeated until the change is <= %lf\n", epsilon);
    diff = epsilon;
    //
    //  Read OUTPUT_FILE from the command line or the user.
    //
    success = sscanf(argv[2], "%s", output_filename);
    if (success != 1)
    {
        printf("\n");
        printf("HEATED_PLATE\n");
        printf(" Error en la lectura del nombre del fichero de salida\n");
        return 1;
    }

    printf("  The steady state solution will be written to %s\n", output_filename);

    initialize_grid(w);

    calculate_solution_gold(w, epsilon, diff);

    //  Write the solution to the output file.
    output = fopen(output_filename, "wt");

    fprintf(output, "%d\n", M);
    fprintf(output, "%d\n", N);

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(output, "%lg ", w[i][j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);

    printf("\n");
    printf(" Solucion escrita en el fichero %s\n", output_filename);
    //
    //  Terminate.
    //
    printf("\n");
    printf("HEATED_PLATE_Serie:\n");
    printf("  Normal end of execution.\n");

    return 0;
}

void initialize_grid(double w[M][N])
{
    int i;
    int j;
    double mean;

    //
    //  Set the boundary values, which don't change.
    //
    for (i = 1; i < M - 1; i++)
        w[i][0] = 100.0;

    for (i = 1; i < M - 1; i++)
        w[i][N - 1] = 100.0;

    for (j = 0; j < N; j++)
        w[M - 1][j] = 100.0;

    for (j = 0; j < N; j++)
        w[0][j] = 0.0;

    //  Average the boundary values, to come up with a reasonable
    //  initial value for the interior.
    mean = 0.0;
    for (i = 1; i < M - 1; i++)
        mean = mean + w[i][0];

    for (i = 1; i < M - 1; i++)
        mean = mean + w[i][N - 1];

    for (j = 0; j < N; j++)
        mean = mean + w[M - 1][j];

    for (j = 0; j < N; j++)
        mean = mean + w[0][j];

    mean = mean / (double)(2 * M + 2 * N - 4);

    printf("\n");
    printf("  MEAN = %lf\n", mean);

    //  Initialize the interior solution to the mean value.

    for (i = 1; i < M - 1; i++)
        for (j = 1; j < N - 1; j++)
            w[i][j] = mean;
}

#undef M
#undef N