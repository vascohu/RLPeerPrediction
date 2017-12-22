#include "InferFunctionsSC.h"
#include "SFMT\SFMT.h"
#include <stdlib.h>
#include <math.h>
#define worker_num_t (worker_num+1)


int task_num;
int worker_num;

int sample_num = 200;
int burn_num = 100;
int interval = 5;

sfmt_t sfmt;

int *y;
int *alpha;
int *beta;

double *yDist;
double *pVec;


void init_class(int _task_num, int _worker_num, double *_yDist, double *_pVec)
{
	task_num = _task_num;
	worker_num = _worker_num;
	yDist = _yDist;
	pVec = _pVec;

	sfmt_init_gen_rand(&sfmt, 1234);

	y = (int *) malloc(2 * task_num * sizeof(int));
	alpha = (int *) malloc(2 * worker_num * sizeof(int));
	beta = (int *) malloc(2 * sizeof(int));
}

int random_sample(double p)
{
	double r = sfmt_genrand_res53(&sfmt);
	if(r<p)
	{
		return(1);
	}
	else
	{
		return(2);
	}
}


void init_y_alpha_beta(int *pLabelMat)
{
	for(int i=0; i<task_num; ++i)
	{
		beta[i*2] = 1;
		beta[i*2+1] = 1;
	}
	for(int j=0; j<worker_num; ++j)
	{
		alpha[j*2] = 2;
		alpha[j*2+1] = 1;
	}
	for(int i= 0; i<task_num; ++i)
	{
		int *votes = (int *) malloc(2 * sizeof(int));
		for(int k=0; k<2; ++k)
		{
			votes[k] = 0;
		}
		int sum_votes = 0;
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
		y[i] = L;
		beta[L]++;
		for(int j=0; j<worker_num; ++j)
		{
			int g = pLabelMat[i*worker_num_t+j]-1;
			if(g==L)
			{
				alpha[j*2]+=1;
			}
			else
			{
				alpha[j*2+1]+=1;
			}
		}
	}
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
	if(L==1)
	{
		z=1;
		w=0;
	}
	else
	{
		z=0;
		w=1;
	}
	return((double)(x+1+z) * (double)(y+w) / (double)(x+1+w) / (double)(y+z));
}

void generate_one_label(int *pLabelMat, int i)
{
	double lambda = 1.0;
	for(int j=0; j<worker_num; j++)
	{
		int L = pLabelMat[i*worker_num+j]-1;
		if(L>=0)
		{
			int x1=0,x2=0;
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
	gs_burn_in(pLabelMat);
	
	for(int i=0; i<2*task_num; ++i)
	{
		yDist[i] = 0.0;
	}

	for(int j=0; j<2*worker_num; ++j)
	{
		pVec[j] = 0.0;
	}

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
		}

		for(int j=0; j<worker_num; ++j)
		{
			pVec[j] += alpha[2*j]/(alpha[2*j]+alpha[2*j+1])/sample_num;
		}
	}

	double accuracy = 0.0;
	for(int i=0; i<task_num; ++i)
	{
		double p = (yDist[2*i]>=yDist[2*i+1])?yDist[2*i]:yDist[2*i+1];
		accuracy += p;
	}
	return(accuracy/task_num);
}