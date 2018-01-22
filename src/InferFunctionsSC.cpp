#include "InferFunctionsSC.h"
#include "SFMT/SFMT.h"
#include <stdlib.h>
#include <math.h>


int task_num;
int worker_num;

int sample_num = 2000;
int burn_num = 1000;
int interval = 5;

sfmt_t sfmt;

int *y;
int *alpha;
int *beta;

double *yDist;
double *pVec;
double *xVec;
double p0;


void init_class(int _task_num, int _worker_num, double *_yDist, double *_pVec, double *_xVec)
{
	task_num = _task_num;
	worker_num = _worker_num;
	yDist = _yDist;
	pVec = _pVec;
	xVec = _xVec;

	sfmt_init_gen_rand(&sfmt, 1234);

	y = (int *) malloc(task_num * sizeof(int));
	alpha = new int[2*worker_num]; //(int *) malloc(2 * worker_num * sizeof(int));
	beta = new int[2]; //(int *) malloc(2 * sizeof(int));
}

int random_sample(double p)
{
	double r = sfmt_genrand_res53(&sfmt);
	if(r<p)
	{
		return(0);
	}
	else
	{
		return(1);
	}
}

int argmax(int *p)
{
	return(p[0]>p[1]?0:1);
}


void init_y_alpha_beta(int *pLabelMat)
{
	beta[0] = 1;
	beta[1] = 1;
	for(int j=0; j<worker_num; ++j)
	{
		alpha[j*2] = 2;
		alpha[j*2+1] = 1;
	}

	int *votes = (int *) malloc(2 * sizeof(int));
	for(int i= 0; i<task_num; ++i)
	{
		int sum_votes = 0;
		for(int k=0; k<2; ++k)
		{
			votes[k] = 1;
			sum_votes += 1;
		}
		for(int j=i*worker_num; j<(i+1)*worker_num; ++j)
		{
			int label = pLabelMat[j]-1;
			if(label>=0)
			{
				votes[label]++;
				sum_votes++;
			}
		}
		int L = random_sample((double)votes[0]/sum_votes);
		//int L = argmax(votes);
		y[i] = L;
		beta[L]++;
		for(int j=0; j<worker_num; ++j)
		{
			int g = pLabelMat[i*worker_num+j]-1;
			if(g==L)
			{
				alpha[j*2]+=1;
			}
			else if(g==1-L)
			{
				alpha[j*2+1]+=1;
			}
		}
	}
	delete(votes);
	/*
	for(int j=0; j<worker_num; ++j)
	{
		printf("%d, %d\n",alpha[2*j], alpha[2*j+1]);
	}
	printf("%d, %d\n", beta[0], beta[1]);
	*/
}

void update_alpha_beta_y(int *pLabelMat, int i, int yn)
{
	int y0 = y[i];
	if(y0!=yn)
	{
		beta[y0]-=1;
		beta[yn]+=1;
		for(int j=0; j<worker_num; ++j)
		{
			int g = pLabelMat[i*worker_num+j]-1;
			if(g==yn)
			{
				alpha[2*j]++;
				alpha[2*j+1]--;
			}
			else if(g==y0)
			{
				alpha[2*j]--;
				alpha[2*j+1]++;
			}
		}
		y[i] = yn;
	}
}

double prob_ratio(int x, int y, int L)
{
	int z,w;
	if(L==0)
	{
		return ((double)x/(double)y);
	}
	else
	{
		return((double)y/(double)x);
	}
}

double calc_lambda(int *pLabelMat, int i)
{
	double lambda = 1.0;
	int x1=0,x2=0;
	if(y[i]==0)
	{
		x1 = beta[0]-1;
		x2 = beta[1];
	}
	else
	{
		x1 = beta[0];
		x2 = beta[1]-1;			
	}
	lambda*=(double)x1/(double)x2;
	for(int j=0; j<worker_num; j++)
	{
		int L = pLabelMat[i*worker_num+j]-1;
		if(L>=0)
		{
			if(L==y[i])
			{
				x1=alpha[2*j]-1;
				x2=alpha[2*j+1];
			}
			else
			{
				x1=alpha[2*j];
				x2=alpha[2*j+1]-1;
			}
			lambda*=prob_ratio(x1, x2, L);
		}
	}
	return(lambda);
}

void generate_one_label(int *pLabelMat, int i)
{

	/*printf("%f, ",lambda);
	if(i==worker_num-1)
	{
		printf("\n");
	}*/
	double lambda = calc_lambda(pLabelMat, i);
	int z = random_sample(lambda/(1+lambda));
	update_alpha_beta_y(pLabelMat, i, z);
}


void gs_burn_in(int *pLabelMat)
{
	init_y_alpha_beta(pLabelMat);
	for(int t=0; t<burn_num; ++t)
	{
		for(int i=0; i<task_num; ++i)
		{
			generate_one_label(pLabelMat, i);
		}
	}
}


double infer(int *pLabelMat)
{

	double accuracy = 0.0;
	gs_burn_in(pLabelMat);
	
	for(int i=0; i<2*task_num; ++i)
	{
		yDist[i] = 0.0;
	}

	for(int i=0; i<task_num; ++i)
	{
		xVec[i] = 0.0;
	}

	for(int j=0; j<worker_num; ++j)
	{
		pVec[j] = 0.0;
	}

	p0 = 0;

	for(int t=0; t<sample_num; ++t)
	{
		for(int n=0; n<interval; ++n)
		{
			for(int i=0; i<task_num; ++i)
			{
				generate_one_label(pLabelMat, i);
			}
		}

		for(int i=0; i<task_num; ++i)
		{
			yDist[2*i+y[i]]+=1.0/sample_num;
			//double lambda = calc_lambda(pLabelMat, i);
			// xVec[i] += log(lambda)/sample_num;
		}


		p0 += (double)beta[0]/(beta[0]+beta[1])/sample_num;

		for(int j=0; j<worker_num; ++j)
		{
			pVec[j] += (double)alpha[2*j]/(alpha[2*j]+alpha[2*j+1])/sample_num;
		}
	}

	//printf("\n%d\t%d\t%f\n", beta[0], beta[1], p0);
	for(int i=0; i<task_num; ++i)
	{
		xVec[i] = log(p0/(1-p0));
		for(int j=0; j<worker_num; ++j)
		{
			if(pLabelMat[i*worker_num+j]==1)
			{
				xVec[i] += log(pVec[j]/(1-pVec[j]));
			}
			else if(pLabelMat[i*worker_num+j]==2)
			{
				xVec[i] += log((1-pVec[j])/pVec[j]);
			}
		}
	}

	for(int i=0; i<task_num; ++i)
	{
		double p = (yDist[2*i]>=yDist[2*i+1])?yDist[2*i]:yDist[2*i+1];
		accuracy += p;
	}
	return(accuracy/task_num);
}

void deconstruct()
{
	delete(y);
	delete(alpha);
	delete(beta);
}
