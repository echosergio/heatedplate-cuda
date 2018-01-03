#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

double cpu_time();

#define M 50
#define N 50

void calculate_solution_gold(double w[M][N], double epsilon, double diff)
{
    double ctime;
    double ctime1;
    double ctime2;
    int i;
    int j;
    int iterations;
    int iterations_print;

    double u[M][N];

    //  iterate until the  new solution W differs from the old solution U
    //  by no more than EPSILON.

    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    ctime1 = cpu_time();

    while (epsilon <= diff)
    {
        //  Save the old solution in U.

        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                u[i][j] = w[i][j];

        //  Determine the new estimate of the solution at the interior points.
        //  The new solution W is the average of north, south, east and west neighbors.

        diff = 0.0;
        for (i = 1; i < M - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;

                if (diff < fabs(w[i][j] - u[i][j]))
                    diff = fabs(w[i][j] - u[i][j]);
            }
        }

        iterations++;
        if (iterations == iterations_print)
        {
            printf("  %8d  %lg\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    } //fin while epsilon

    ctime2 = cpu_time();
    ctime = ctime2 - ctime1;

    printf("\n");
    printf("  %8d  %lg\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  CPU time = %f\n", ctime);
}

double cpu_time() 
{
    double value;

    value = (double)clock() / (double)CLOCKS_PER_SEC;

    return value;
}

#undef M
#undef N