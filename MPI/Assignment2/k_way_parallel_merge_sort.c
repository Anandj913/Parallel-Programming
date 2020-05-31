/*
Name: Anand Jhunjhunwala, Sombit Dey
Roll Number: 17EC30041, 17EC10056
Assignment number: 3
|----MPI Assignment----|
Topic: (k=2) Parallel Merge Sort

Compilation Instruction 
Run from the terminal: mpicc 17EC30041_17EC10056.c -lm
Then after successfull compilation run: mpirun -np NUMBER_OF_PROCESS_TO_RUN ./a.out
*/

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>



int* merge_parallel(int *a, int* b , int size)
{
    int i = 0, j =0;
    int *temp = (int *)malloc(size*sizeof(int));
    int k = 0;
    while(i < size/2 && j< size-size/2)
    {
        if(a[i] <= b[j])
        {
            temp[k++] = a[i++];
        }
        else
        {
            temp[k++] = b[j++];
        }
    }
    while(i<size/2)
    {
        temp[k++] = a[i++];
    }
    while(j < size - size/2)
    {
        temp[k++] = b[j++];
    }
   return temp;
}
// int binarySearch(int key, int * a , int size)
// {
//     int low = 0;
//     int high = size-1;
//     int mid = 0;
//     while (low <= high)
//     {
//         mid = (low + high) / 2;
//         if ( a[mid]  < key)
//         {
//             low = mid + 1;
//         }
//         else if (a[mid]  > key)
//         {
//             high = mid - 1;
//         }
//         else
//         {
//             return mid;
//         }
//     }
//     return mid;
// }
int my_binary_search( int value, int* a, int left, int right )
{
    int low  = left;
    int high = left;
    if(left<right+1)
    	high = right+1;
    while( low < high )
    {
        int mid = ( low + high ) / 2;
        if ( value <= a[ mid ] ) high = mid;
        else low  = mid + 1; // because we compared to a[mid] and the value was larger than a[mid].
                             // Thus, the next array element to the right from mid is the next possible
                             // candidate for low, and a[mid] can not possibly be that candidate.
    }
 return high;
}
// void exchange( int &A, int &B )
// {
//     int t = A;
//     A = B;
//     B = t;
// }
// // Merge two ranges of source array T[ p1 .. r1 ] and T[ p2 .. r2 ] into destination array A starting at index p3.
// From 3rd ed. of "Introduction to Algorithms" p. 798-802
// Listing 2 (which also needs to include the binary search implementation as well)
void merge_dac( int * t, int p1, int r1, int p2, int r2, int * a,int p3  )
{	
	
    int length1 = r1 - p1 + 1;
    int length2 = r2 - p2 + 1;
    // if ( length1 < length2 )
    // {
    //     exchange(      p1,      p2 );
    //     exchange(      r1,      r2 );
    //     exchange( length1, length2 );
    // }
    if ( length1 == 0 ) return;
    int q1 = ( p1 + r1 ) / 2;
    int q2 = my_binary_search( t[ q1 ], t, p2, r2 );
    int q3 = p3 + ( q1 - p1 ) + ( q2 - p2 );
    a[ q3 ] = t[ q1 ];
    merge_dac( t, p1,     q1 - 1, p2, q2 - 1, a, p3     );
    merge_dac( t, q1 + 1, r1,     q2, r2,     a, q3 + 1 );
}
int main(int argc, char **argv)
{
    int total = pow(10, 2);
    MPI_Init(&argc, &argv);
    int comm_rank, comm_node, i;
    int *unsorted_arr = NULL, *sorted_arr = NULL, *list1 = NULL, *list2 = NULL;
    int m = pow(2,16) -1, a = 72;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_node);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Group subgrp1 = MPI_GROUP_EMPTY, subgrp2 = MPI_GROUP_EMPTY;
    MPI_Comm grp1_comm = MPI_COMM_NULL, grp2_comm = MPI_COMM_NULL;
    void merge(int *a, int size);
    void serial_mergesort(int *a, int size);
    void non_root_node(int rank, int max_rank, int tag, MPI_Comm comm);
    void root_node(int *a, int size, int max_rank, int tag, MPI_Comm comm);
    void parallel_mergesort(int *a, int size, int level, int rank, int max_rank, int tag, MPI_Comm comm);
    int top_level(int rank);
    int color;
    int tag = 23; 
    color = comm_rank/(comm_node/2);
    if(color == 0)
    {
     MPI_Comm_split(MPI_COMM_WORLD, color, comm_rank, &grp1_comm);
    }
    else
    {
     MPI_Comm_split(MPI_COMM_WORLD, color, comm_rank, &grp2_comm);  
    }
    if(grp1_comm != MPI_COMM_NULL)
    {
        MPI_Comm_group(grp1_comm, &subgrp1);
    }
    else
    {
        MPI_Comm_group(grp2_comm, &subgrp2);
    }

    int grp1_rank= -1, grp1_node=-1, grp2_rank= -1, grp2_node=-1;
    int max_rank;
    if(grp1_comm != MPI_COMM_NULL)
    {
        MPI_Comm_size(grp1_comm, &grp1_node);
    	MPI_Comm_rank(grp1_comm, &grp1_rank);
    	max_rank = grp1_node-1;
    }
    else
    {
        MPI_Comm_size(grp2_comm, &grp2_node);
    	MPI_Comm_rank(grp2_comm, &grp2_rank);
    	max_rank = grp2_node-1;
    }

    if (grp1_rank == 0)
    {
        unsorted_arr = (int *)malloc(total*sizeof(int));
        unsorted_arr[0] = 1;
        printf("\n|-----Generating Random number at process:%d-----|\n", grp1_rank);
        for(i=1;i<total;i++)
        {
            unsorted_arr[i] = (a*unsorted_arr[i-1])%m;
        }
        list1 = (int *)malloc((total/2)*sizeof(int));
        list2 = (int *)malloc((total-total/2)*sizeof(int));
        memcpy(list1, unsorted_arr, (total/2)*sizeof(int));
        memcpy(list2, unsorted_arr + total/2, (total-total/2)*sizeof(int));
        //send list 2 and start working on list1 in group1
        MPI_Request request; 
        MPI_Status status;
        printf("\n|-----Sending List 2 for Sorting to group 2-----|\n");
        MPI_Isend(list2, total - total/2, MPI_INT, comm_node/2, 0, MPI_COMM_WORLD, &request);
        printf("\n|-----Sorting List 1-----|\n");
        root_node(list1, total/2, max_rank, tag, grp1_comm);
        MPI_Wait(&request, &status);
        MPI_Recv(list2, total - total/2, MPI_INT, comm_node/2, tag, MPI_COMM_WORLD, &status);
        printf("\n|-----Received sorted List 2-----|\n");
        printf("\n|-----Merging List 1 and List 2-----|\n");
        sorted_arr = merge_parallel(list1,list2,total);
        printf("\n|-----Sorted Array:-----| \n");
        for (int i = 0; i < total; ++i)
        {
        	printf("%d ",sorted_arr[i] );
        	if((i+1)%10 == 0)
        		printf("\n");
        }
        printf("\n|--------------------End--------------------|\n");
    }
    else if(grp2_rank == -1)
    {
    	non_root_node(grp1_rank, max_rank, tag, grp1_comm);
    }

    if( comm_rank == comm_node/2)
    {
        //receive list 2 and start working on it in group2 
        list2 = (int *)malloc((total-total/2)*sizeof(int));
        MPI_Recv(list2, total-total/2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\n|-----Received List 2 for Sorting to group 2-----|\n");
        printf("\n|-----Sorting List 2-----|\n");
        root_node(list2, total-total/2, max_rank, tag, grp2_comm);
        printf("\n|-----Sent Sorted List 2 to group 1-----|\n");
        MPI_Send(list2, total-total/2, MPI_INT, 0, tag, MPI_COMM_WORLD);


    }
    else if(grp1_rank == -1)
    {
    	non_root_node(grp2_rank, max_rank, tag, grp2_comm);
    }
   
    MPI_Finalize();
}

int top_level(int rank)
{
	int level = 0;
	while(pow(2,level) <= rank)
	{
		level++;
	}
	return level;
}
void merge(int *a, int size)
{
    int i = 0, j = size/2;
    int *temp = (int *)malloc(size*sizeof(int));
    int k = 0;
    while(i < size/2 && j< size)
    {
        if(a[i] <= a[j])
        {
            temp[k++] = a[i++];
        }
        else
        {
            temp[k++] = a[j++];
        }
    }
    while(i<size/2)
    {
        temp[k++] = a[i++];
    }
    while(j < size)
    {
        temp[k++] = a[j++];
    }
    memcpy(a, temp, size*sizeof(int));
    free(temp);
}

void serial_mergesort(int *a, int size)
{
    if(size <= 1)
    {
        return;
    }
    else
    {
        serial_mergesort(a, size/2);
        serial_mergesort(a + size/2, size-size/2);
        merge(a, size);
    }
    return;
}
void parallel_mergesort(int *a, int size, int level, int rank, int max_rank, int tag, MPI_Comm comm)
{
	int new_rank = rank + pow(2, level);
	if(new_rank > max_rank)
	{
		serial_mergesort(a, size);
	}
	else
	{
		MPI_Request request;
		//send next half for sorting 
		MPI_Isend(a + size/2, size-size/2, MPI_INT, new_rank, tag, comm, &request);
		//sort first half
		parallel_mergesort(a, size/2, level+1, rank, max_rank, tag, comm);
		MPI_Request_free(&request);
		//receive next sorted half
		MPI_Recv(a + size/2, size-size/2, MPI_INT, new_rank, tag, comm, MPI_STATUS_IGNORE);
		merge(a, size);
	}
	return;
}
void non_root_node(int rank, int max_rank, int tag, MPI_Comm comm)
{
	int level = top_level(rank);
	MPI_Status status;
	int size;
	MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
	MPI_Get_count(&status, MPI_INT, &size);
	int source_rank = status.MPI_SOURCE;
	int *a = malloc(size*sizeof(int));
	//receive other half
	MPI_Recv(a, size, MPI_INT, source_rank, tag, comm, &status);
	//sort other half
	parallel_mergesort(a,size,level,rank,max_rank,tag,comm);
	//send other half
	MPI_Send(a, size, MPI_INT, source_rank, tag, comm);

}
void root_node(int *a, int size, int max_rank, int tag, MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);
	parallel_mergesort(a, size, 0, rank, max_rank, tag, comm);
}