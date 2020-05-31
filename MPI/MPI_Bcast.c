#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>


int get_loop(int a)
{
int i =0;
while(a/pow(2,i))
++i;
return i;
}

int main ()
{
  int num_elements;
  MPI_Init(NULL, NULL);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
 int rank = world_rank;
  int loop_no= get_loop(rank);
  int total_loop = log2(world_size);
  total_loop++;
  // int data = 5;
  // int i=0;
  double total_my_bcast_time = 0.0;
  double total_mpi_bcast_time = 0.0;
  int i, data;
  if(world_rank == 0)
  {
    data =5;
  }

    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time -= MPI_Wtime();
  while(i < (total_loop-loop_no))
  {
 
    if(i>=loop_no)
      {
        if((pow(2,i) + rank )<world_size)
        MPI_Send(&data, 1, MPI_INT, pow(2,i)+ rank, 10, MPI_COMM_WORLD);
     
      }
    else
    {
        MPI_Recv(&data, 1, MPI_INT, rank%2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    i++;
  }

    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time += MPI_Wtime();

    // Time MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time -= MPI_Wtime();
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time += MPI_Wtime();

  // Print off timing information
  if (world_rank == 0) 
  {
    printf("Data size = %d\n", num_elements * (int)sizeof(int));
    printf("Avg my_bcast time = %lf\n", total_my_bcast_time );
    printf("Avg MPI_Bcast time = %lf\n", total_mpi_bcast_time);
  }

  MPI_Finalize();
}