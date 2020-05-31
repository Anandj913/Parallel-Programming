/*
Name: Anand Jhunjhunwala
Roll Number: 17EC30041
Assignment number: 1
|----MPI Assignment----|
Topic: Histogram equilization and edge detection using sobel filter

Compilation Instruction 
Put the image file in the same folder as of code
Run from the terminal: mpicc 17EC30041_Assignment2.c -lpng -lm
Then after successfull compilation run: mpirun -np NUMBER_OF_PROCESS_TO_RUN ./a.out
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <png.h>
#include <math.h>

const char *img = "sample1.png";
char *hist_name = "histeql.pgm";
char *filter_name = "final.pgm";



static png_structp png_ptr = NULL;
static png_infop info_ptr = NULL;

png_uint_32  width, height;
int  bit_depth, color_type, interlace_method, compression_method, filter_method;
png_bytepp image;

void image_writer(char *filename, int *data, int height, int width)
{
	int i,j;
	FILE* pgmimg;
   	pgmimg = fopen(filename, "w"); //write the file in binary mode
   	fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
   	fprintf(pgmimg, "%d %d\n", (int)width, (int)height); // Writing Width and Height into the file
   	fprintf(pgmimg, "255\n"); // Writing the maximum gray value
   	for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
         fprintf(pgmimg, "%d ", data[i*width + j]); //Copy gray value from array to file
      }
      fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv)
{
	int param[3];
	int rank, node, partition, extra,i,j; 
	int *partition_ar, *fimage, *section, *histimg, *filter_img;
	long int *hist, *fhist;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &node);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	hist = (long int *)calloc(256, sizeof(long int));
	fhist = (long int *)calloc(256, sizeof(long int));
	if(rank == 0)
	{
		printf("\n|---------------|Running code on %d processor|---------------|\n", node);
		printf("\nReading Image from process: %d", rank);
		FILE *IMG;
		partition_ar = (int *)calloc(node, sizeof(int));
		IMG = fopen(img, "rb");
		png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		info_ptr = png_create_info_struct (png_ptr);
		png_init_io (png_ptr, IMG);
	    png_read_png (png_ptr, info_ptr, 0, 0);
	    png_get_IHDR (png_ptr, info_ptr, & width, & height, & bit_depth,
			  & color_type, & interlace_method, & compression_method,
			  & filter_method);
	    image = png_get_rows(png_ptr, info_ptr);
	    fimage = (int *)malloc(height*width*sizeof(int));
		printf("\nHeight of image: %ld | Width of image: %ld", height, width);
	    //linearize_image
	    for(i=0;i<height*width;i++)
	    {
	    	fimage[i] = image[i/width][i%width];
	    }
	    // deciding the partition of image
		printf("\nCreating partition of image");
	    partition = (int)(height/node);
	    extra = height%node;
	    for(i=node-1; i>=0; i--)
	    {
	    	partition_ar[i] = partition;
	    	if(extra >0)
	    	{
	    		partition_ar[i]++;
	    		extra--;
	    	}
	    }
		// printf("\nPartition array from process: %d\n", rank);
		// for(i=0; i<node; i++)
		// {
		// 	printf("%d ", partition_ar[i]);
		// }
	    param[0] = width;
	    param[2] = width*height;
	    for(i = 1; i< node; i++)
	    {
	    	param[1] = partition_ar[i];
			//printf("\nSending param_value to process: %d", i);
	    	MPI_Send(param, 3, MPI_INT, i , i, MPI_COMM_WORLD);
	    }
	    printf("\n|---All param values sent---|");
	    param[1] = partition_ar[0];
	    int gap = partition_ar[0];

	    for(i=1; i<node; i++)
	    {
	    	section = (int *)malloc(width*(partition_ar[i]+2)*sizeof(int));
	    	if(i != node-1)
	    	{
		    	memcpy(section, fimage + gap*width, width*(partition_ar[i]+2)*sizeof(int));
		    	//printf("\nSending image_section to process: %d", i);
		    	MPI_Send(section, (partition_ar[i]+2)*width, MPI_INT, i, i, MPI_COMM_WORLD);
	    	}
	    	else
	    	{
		    	memcpy(section, fimage + gap*width, width*(partition_ar[i])*sizeof(int));
		    	//printf("\nSending image_section to process: %d", i);
		    	MPI_Send(section, (partition_ar[i])*width, MPI_INT, i, i, MPI_COMM_WORLD);
	    	}
	    	free(section);
	    	gap = gap + partition_ar[i];
	    }
	    printf("\n|---All image sections sent---|");
	    section = (int *)malloc(width*(partition_ar[0]+2)*sizeof(int));
	    memcpy(section, fimage, width*(partition_ar[0]+2)*sizeof(int));

	}
	else //if rank != 0 
	{
		//printf("\nReceving param to process: %d", rank);
		MPI_Recv(param, 3, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if(rank != node-1)
		{
			section = (int *)malloc((param[1]+2)*param[0]*sizeof(int));
			MPI_Recv(section,(param[1]+2)*param[0], MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		   // printf("\nReceived image_section to process: %d", rank);
		}
		else
		{
			section = (int *)malloc((param[1])*param[0]*sizeof(int));
			MPI_Recv(section,(param[1])*param[0], MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		   // printf("\nReceived image_section to process: %d", rank);
		}
	}
	if(rank == 0)
	{
		printf("\n\n|-----Running histogram equilization code-----|\n");
		printf("\n|---Processing individual image sections---|");
	}

	// all image section received along with width and height stored in param[0] and param[1]

	/* ------------------------- histogram equilization -------------------------------------- */
	for(i=0; i<param[0]*param[1]; i++)
	{
		hist[section[i]] = hist[section[i]] + 1;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(hist, fhist, 256, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	//fhist contain final histogram
	histimg = (int *)calloc((param[1]+2)*param[0], sizeof(int));
	long int sum = param[2];
	long int preval = 0;
	long int newval = 0;
	for(i = 255; i>=0; i--)
		{
			sum = sum - preval;
			newval = sum;
			preval = fhist[i];
			newval = newval*255;
			newval = abs(newval/param[2]);
			fhist[i] = newval;
		}
	if(rank != node-1)
	{
		for(i =0; i<(param[0]*(param[1]+2)); i++)
		{
			histimg[i] = fhist[section[i]];
		}
	}
	else
	{
		for(i =0; i<(param[0]*(param[1])); i++)
		{
			histimg[i] = fhist[section[i]];
		}
	}
	// histimg is the equilied img, sending the equilized image
	if(rank != 0)
	{
		//printf("\nSending hist_img from process: %d", rank);
		MPI_Send(histimg, param[0]*(param[1]+2), MPI_INT, 0, rank, MPI_COMM_WORLD);
	}
	else
	{

		int *final_histogram_img, *rec_buff;
		final_histogram_img = (int *)malloc(height*width*sizeof(int));
		int gap = partition_ar[0];
		memcpy(final_histogram_img, histimg, partition_ar[0]*width*sizeof(int));
		for(i = 1; i<node; i++)
		{
			rec_buff = (int *)malloc(sizeof(int)*width*(partition_ar[i]+2));
			MPI_Recv(rec_buff, width*(partition_ar[i]+2), MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    //printf("\nReceived hist_img from process: %d", i);
			memcpy(final_histogram_img + gap*width, rec_buff, partition_ar[i]*width*sizeof(int));
			gap = gap + partition_ar[i];
			free(rec_buff);
		}
		printf("\n|---Received all processed image section---|");
		printf("\n|---Writing Histogram equilized image---|\n");
		image_writer(hist_name, final_histogram_img, height, width);
		printf("\n|---Histogram equilized image written---|\n");


	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0)
	{
		printf("\n\n|-----Running sobel filter code-----|\n");
		printf("\n|---Processing individual image sections---|");

	}
	/*--------------------- sobel filter code -------------------------------*/
	int gradient_h , gradient_v; 
	float gradient; 
	int THRESH = 15; //threshold value for sobel filter
	filter_img = (int *)calloc((param[1]+2)*param[0], sizeof(int));
	for(i = param[0]; i<param[0]*(param[1]+1); i++)
	{

		if(i%param[0] == 0)
		{
			gradient_v = 2*histimg[i+param[0]] + histimg[i+1+param[0]] - 2*histimg[i-param[0]] - histimg[i+1-param[0]];
			gradient_h = histimg[i+param[0]+1] + 2*histimg[i+1] + histimg[i-param[0]+1];
			gradient = sqrt((gradient_v * gradient_v) + (gradient_h * gradient_h));
			if(gradient < THRESH)
			{
				gradient = 0;
			}
			else
			{
				gradient = 255;
			}
			filter_img[i] = (int)gradient; 
		}
		if(i%param[0] == param[0]-1)
		{
			gradient_v = 2*histimg[i+param[0]] + histimg[i -1 +param[0]] - 2*histimg[i-param[0]] - histimg[i-1-param[0]];
			gradient_h = histimg[i+param[0]-1] + 2*histimg[i-1] + histimg[i-param[0]-1];
			gradient = sqrt((gradient_v * gradient_v) + (gradient_h * gradient_h));
			if(gradient < THRESH)
			{
				gradient = 0;
			}
			else
			{
				gradient = 255;
			}
			filter_img[i] = (int)gradient; 
		}
		else
		{
			gradient_v = 2*histimg[i+param[0]] + histimg[i +1 +param[0]] + histimg[i -1 +param[0]]- 2*histimg[i-param[0]] - histimg[i-1-param[0]] - histimg[i-param[0] +1];
			gradient_h = histimg[i+param[0]+1] + 2*histimg[i+1] + histimg[i-param[0]+1] - histimg[i+param[0]-1] - 2*histimg[i-1] - histimg[i-param[0]-1];
			gradient = sqrt((gradient_v * gradient_v) + (gradient_h * gradient_h)); 
			if(gradient < THRESH)
			{
				gradient = 0;
			}
			else
			{
				gradient = 255;
			}
			filter_img[i] = (int)gradient; 
		}
	}
	if(rank != 0)
	{
		//printf("\nSending filtered_image from process: %d", rank);
		MPI_Send(filter_img, param[0]*(param[1]+2), MPI_INT, 0, rank, MPI_COMM_WORLD);
	}
	else
	{
		int *final_filter_img, *recvf_buff;
		final_filter_img = (int *)malloc(height*width*sizeof(int));
		memcpy(final_filter_img, section, width*sizeof(int)); // copy 1st row
		memcpy(final_filter_img + width, filter_img + width, partition_ar[0]*width*sizeof(int));
		int gap = partition_ar[0];
		for(i = 1; i<node-1; i++)
		{
			recvf_buff = (int *)malloc(sizeof(int)*width*(partition_ar[i]+2));
			MPI_Recv(recvf_buff, width*(partition_ar[i]+2), MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//printf("\nReceived filtered_image from process: %d", i);
			memcpy(final_filter_img + gap*width + width, recvf_buff + width, partition_ar[i]*width*sizeof(int));
			gap = gap + partition_ar[i];
			free(recvf_buff);
		}
		recvf_buff = (int *)malloc(sizeof(int)*width*(partition_ar[i]+2));
		MPI_Recv(recvf_buff, width*(partition_ar[i]+2), MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("\nReceived filtered_image from process: %d", node-1);
		memcpy(final_filter_img + gap*width + width, recvf_buff + width, (partition_ar[i]-1)*width*sizeof(int));
		gap = gap + partition_ar[i];
		free(recvf_buff);
		printf("\n|---Received all processed image section---|");
		printf("\n|---Writing Filtered image---|\n");
		image_writer(filter_name, final_filter_img, height, width);	
		printf("\n|---Filtered image written successfully---|\n");
		printf("\n\n|---------------------------|End|---------------------------|\n");

	}
	MPI_Finalize();
	return 0;
}