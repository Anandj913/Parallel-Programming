/*
-------------------------------------------------------------------------------
Name: Anand Jhunjhunwala
Roll Number: 17EC30041
Assignment 1: Rotation about an arbitrary axis in 3D
-------------------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h> 
#include<math.h>
#include <iostream>
using namespace std;
#define Max_Object_size  10000
#define PI 3.14159265
typedef struct {
    float x;
    float y;
    float z;
}Point;

Point P, Q;
Point Object[Max_Object_size];
Point Object_ans[Max_Object_size];

float **translation;
float **Rotation_about_x;
float **Rotation_about_y;
float **Rotation_about_z;
float **Inverse_rotation_about_y; 
float **Inverse_rotation_about_x;
float **Inverse_translation; 

float ** Transpose_of_matrix(float **A)
{
	float **B;
	int i,j;
	B = (float **)malloc(4*sizeof(float *));
	for(i=0; i<4; i++)
	{
		B[i] = (float *)malloc(4*sizeof(float));
	}
	for (i = 0; i < 4; ++i)
	{
    for (j = 0; j < 4; ++j) {
    	B[j][i] = A[i][j];
    }
  }

  return B; 
}

float ** allocate_memory()
{
	int i;
	float **B;
	B = (float **)malloc(4*sizeof(float *));
	for(i=0; i<4; i++)
	{
		B[i] = (float *)malloc(4*sizeof(float));
	}
	return B;

}

void print_matrix( float **A)
{
	int i,j;
	printf("\nmatrix:\n");
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j) {
            printf("%f  ", A[i][j]);
            if (j == 4 - 1)
                printf("\n");
        }
}


void read_axis(char *filename4arbitraryAxis)
{
	char c[100];
	char *temp[4]; 
	FILE *faxis;
	if ((faxis = fopen(filename4arbitraryAxis, "r")) == NULL)
		{
		  printf("Error! opening file %s", filename4arbitraryAxis);
		  exit(1);
		}
	else
		{
			fscanf(faxis, "%[^\n]", c);
			int i = 0;
			temp[i] = strtok(c, "(");
			
			while( temp[i] != NULL)
			{
				i++;
				temp[i] = strtok(NULL, "(" );

			}
			temp[0] = strtok(temp[1], ")");
			temp[1] = strtok(temp[2], ")");
			char *digitP[4];
			char *digitQ[4];
			i = 0;
			digitP[i] = strtok(temp[0], ",");
			
			while( digitP[i] != NULL)
			{
				i++;
				digitP[i] = strtok(NULL, "," );

			}
			i = 0;
			digitQ[i] = strtok(temp[1], ",");
			
			while( digitQ[i] != NULL)
			{
				i++;
				digitQ[i] = strtok(NULL, "," );

			}
			P.x = atof(digitP[0]);
			P.y = atof(digitP[1]);
			P.z = atof(digitP[2]);
			Q.x = atof(digitQ[0]);
			Q.y = atof(digitQ[1]);
			Q.z = atof(digitQ[2]);
			
			fclose(faxis);
		}
}

int read_object(char *filename4objectFile)
{
	FILE *fobject;
	int i = 0;
	if ((fobject = fopen(filename4objectFile, "r")) == NULL)
		{
		  printf("Error! opening file %s", filename4objectFile);
		  exit(1);
		}
	else
	{
		int buff = 100;
		char c[100];
		i=i+1;
		char *temp[4];
		while(fgets(c, buff, fobject))
		{

			if(i != Max_Object_size)
				{
					int j = 0;
					temp[j] = strtok(c, " ");
					Object[i-1].x = atof(temp[0]);
					while( temp[j] != NULL)
						{
							j++;
							temp[j] = strtok(NULL, " " );

						}
					Object[i-1].y = atof(temp[1]);
					Object[i-1].z = atof(temp[2]);
					//printf("%s\n",c);
					i=i+1;
				}
			else
			{
				break;
			}
		}
	}
	fclose(fobject);
	return i-1;
}

float ** define_x_matrix(Point C)
{
	float b = sqrt(C.y*C.y + C.z*C.z );
	float **B;
	B = allocate_memory();
	B[0][0] = 1;
	B[0][1] = 0;
	B[0][2] = 0;
	B[0][3] = 0;
	B[1][0] = 0;
	B[1][1] = C.z/b;
	B[1][2] = C.y/b;
	B[1][3] = 0;
	B[2][0] = 0;
	B[2][1] = (-1*C.y)/b;
	B[2][2] = C.z/b;
	B[2][3] = 0;
	B[3][0] = 0;
	B[3][1] = 0;
	B[3][2] = 0;
	B[3][3] = 1;
	return B;
}

float ** define_y_matrix(Point C)
{
	float b = sqrt(C.y*C.y + C.z*C.z );
	float **B;
	B = allocate_memory();
	B[0][0] = b;
	B[0][1] = 0;
	B[0][2] = C.x;
	B[0][3] = 0;
	B[1][0] = 0;
	B[1][1] = 1;
	B[1][2] = 0;
	B[1][3] = 0;
	B[2][0] = -1*C.x;
	B[2][1] = 0;
	B[2][2] = b;
	B[2][3] = 0;
	B[3][0] = 0;
	B[3][1] = 0;
	B[3][2] = 0;
	B[3][3] = 1;

	return B;
}

float ** define_z_matrix(float angle)
{
	float **B;
	angle = (angle*PI)/180;
	B = allocate_memory();
	B[0][0] = cos(angle);
	B[0][1] = sin(angle);
	B[0][2] = 0;
	B[0][3] = 0;
	B[1][0] = -1*sin(angle);
	B[1][1] = cos(angle);
	B[1][2] = 0;
	B[1][3] = 0;
	B[2][0] = 0;
	B[2][1] = 0;
	B[2][2] = 1;
	B[2][3] = 0;
	B[3][0] = 0;
	B[3][1] = 0;
	B[3][2] = 0;
	B[3][3] = 1;

	return B;
}

float ** define_translation(Point A)
{
	float ** B;
	B = allocate_memory();

	B[0][0] = 1;
	B[0][1] = 0;
	B[0][2] = 0;
	B[0][3] = -A.x;
	B[1][0] = 0;
	B[1][1] = 1;
	B[1][2] = 0;
	B[1][3] = -A.y;
	B[2][0] = 0;
	B[2][1] = 0;
	B[2][2] = 1;
	B[2][3] = -A.z;
	B[3][0] = 0;
	B[3][1] = 0;
	B[3][2] = 0;
	B[3][3] = 1;
	return B;
}
float ** define_translation_inverse(Point A)
{
	float ** B;
	B = allocate_memory();

	B[0][0] = 1;
	B[0][1] = 0;
	B[0][2] = 0;
	B[0][3] = A.x;
	B[1][0] = 0;
	B[1][1] = 1;
	B[1][2] = 0;
	B[1][3] = A.y;
	B[2][0] = 0;
	B[2][1] = 0;
	B[2][2] = 1;
	B[2][3] = A.z;
	B[3][0] = 0;
	B[3][1] = 0;
	B[3][2] = 0;
	B[3][3] = 1;
	return B;
}

float** matrix_multi(float **a,float **b)
{
	int i,j,k;
	float **c=(float **)malloc(4*sizeof(float *));
	for(i=0;i<4;i++)
	c[i]=(float*)malloc(sizeof(float)); 
#pragma omp parallel shared(a,b,c) private(i,j,k)
{
	#pragma omp for schedule(static)
	for(i=0;i<4;i++)
	{
		for(j=0;j<1;j++)
		c[i][j]=0;
	}

	#pragma omp for schedule(static)	
	for(i=0;i<4;i++)
	{
		for(j=0;j<1;j++)
		{
			for(k=0;k<4;k++)
			c[i][j]+=a[i][k]*b[k][j];
		}
	}
}
return c;
}
int main(int argc, char *argv[])
{

int i,j,k;
Point C;
if (argc < 5 || argc > 6)  
	{ 
	  printf("enter 5 arguments only eg.\"filename number_of_threads filename4arbitraryAxis filename4objectFile angleOFrotation\""); 
	  return 0; 
	} 
int number_of_threads = atoi(argv[1]);  
float angle_of_rotation = atoi(argv[4]); 
read_axis(argv[2]); // Result is stored in P and Q
// printf("%f, %f, %f, %f, %f, %f\n", P.x, P.y, P.z, Q.x, Q.y, Q.z);
int total_object = read_object(argv[3]);
float d = sqrt((Q.x-P.x)*(Q.x-P.x) + (Q.y-P.y)*(Q.y-P.y) + (Q.z-P.z)*(Q.z-P.z));
C.x = (Q.x-P.x)/d;
C.y = (Q.y-P.y)/d;
C.z = (Q.z-P.z)/d;
double time1=omp_get_wtime();

translation = define_translation(P);
Inverse_translation = define_translation_inverse(P);
Rotation_about_x = define_x_matrix(C);
Rotation_about_y = define_y_matrix(C);
Rotation_about_z = define_z_matrix(angle_of_rotation);
Inverse_rotation_about_x = Transpose_of_matrix(Rotation_about_x);
Inverse_rotation_about_y = Transpose_of_matrix(Rotation_about_y);

// float ** temp1,**temp2,**temp3, **temp4,**temp5,**Transformation;
// temp1 = matrix_multi(Rotation_about_x,translation);
// temp2 = matrix_multi(Rotation_about_z,Rotation_about_y);
// temp3 = matrix_multi(Inverse_rotation_about_x,Inverse_rotation_about_y);

// temp4 = matrix_multi(temp2,temp1);
// temp5 = matrix_multi(Inverse_translation,temp3);

// Transformation = matrix_multi(temp5,temp4);

float **B;
B = (float **)malloc(4*sizeof(float *));
for(i=0; i<4; i++)
{
	B[i] = (float *)malloc(1*sizeof(float));
}
	






omp_set_num_threads(number_of_threads);
	#pragma omp parallel 
{
		#pragma omp for
		for(i=0;i<total_object;i++)
		{ 	
			// cout << "thread id " << omp_get_thread_num() << endl;
			B[0][0] = Object[i].x;
			B[1][0] = Object[i].y;
			B[2][0] = Object[i].z;
			B[3][0] = 1;
			B = matrix_multi(translation, B);
			B = matrix_multi(Rotation_about_x, B);
			B = matrix_multi(Rotation_about_y, B);
			B = matrix_multi(Rotation_about_z, B);
			B = matrix_multi(Inverse_rotation_about_y, B);
			B = matrix_multi(Inverse_rotation_about_x, B);
			B = matrix_multi(Inverse_translation, B);

			 Object_ans[i].x = B[0][0];
			 Object_ans[i].y = B[1][0];
			 Object_ans[i].z = B[2][0];
			
		}
}
double time2=omp_get_wtime();
for (i=0;i<total_object;++i){cout << Object_ans[i].x << " " <<Object_ans[i].y <<" " << Object_ans[i].z << endl;}
cout <<"Total objects are  " << total_object <<endl;
cout << "The time for multiplication is " << time2 -time1 <<endl; 

}
