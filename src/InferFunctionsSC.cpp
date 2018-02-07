#include "InferFunctionsSC.h"
#include "SFMT/SFMT.h"
#include <stdlib.h>
#include <math.h>


long task_num;
long worker_num;

long sample_num = 5000;
long burn_num = 1000;
long longerval = 2;

sfmt_t sfmt;

long *y;
long *alpha;
long *beta;
double *pDist;

double *yDist;
double *pVec;
double *xVec;
double p0;


void init_class(long _task_num, long _worker_num, double *_yDist, double *_pVec, double *_xVec, double *_pDist)
{
	task_num = _task_num;
	worker_num = _worker_num;
	yDist = _yDist;
	pVec = _pVec;
	xVec = _xVec;
	pDist = _pDist;

	sfmt_init_gen_rand(&sfmt, 1234);

	y = (long *) malloc(task_num * sizeof(long));
	alpha = new long[2*worker_num]; //(long *) malloc(2 * worker_num * sizeof(long));
	beta = new long[2]; //(long *) malloc(2 * sizeof(long));
}

long random_sample(double p)
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

long argmax(long *p)
{
	return(p[0]>p[1]?0:1);
}


void init_y_alpha_beta(long *pLabelMat)
{
	beta[0] = 1;
	beta[1] = 1;
	for(long j=0; j<worker_num; ++j)
	{
		alpha[j*2] = 2;
		alpha[j*2+1] = 1;
	}

	long *votes = (long *) malloc(2 * sizeof(long));
	for(long i= 0; i<task_num; ++i)
	{
		long sum_votes = 0;
		for(long k=0; k<2; ++k)
		{
			votes[k] = 1;
			sum_votes += 1;
		}
		for(long j=i*worker_num; j<(i+1)*worker_num; ++j)
		{
			long label = pLabelMat[j]-1;
			if(label>=0)
			{
				votes[label]++;
				sum_votes++;
			}
		}
		long L = random_sample((double)votes[0]/sum_votes);
		//long L = argmax(votes);
		y[i] = L;
		beta[L]++;
		for(long j=0; j<worker_num; ++j)
		{
			long g = pLabelMat[i*worker_num+j]-1;
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
	for(long j=0; j<worker_num; ++j)
	{
		prlongf("%d, %d\n",alpha[2*j], alpha[2*j+1]);
	}
	prlongf("%d, %d\n", beta[0], beta[1]);
	*/
}

void update_alpha_beta_y(long *pLabelMat, long i, long yn)
{
	long y0 = y[i];
	if(y0!=yn)
	{
		beta[y0]-=1;
		beta[yn]+=1;
		for(long j=0; j<worker_num; ++j)
		{
			long g = pLabelMat[i*worker_num+j]-1;
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

double prob_ratio(long x, long y, long L)
{
	long z,w;
	if(L==0)
	{
		return ((double)x/(double)y);
	}
	else
	{
		return((double)y/(double)x);
	}
}

double calc_lambda(long *pLabelMat, long i)
{
	double lambda = 1.0;
	long x1=0,x2=0;
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
	for(long j=0; j<worker_num; j++)
	{
		long L = pLabelMat[i*worker_num+j]-1;
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

void generate_one_label(long *pLabelMat, long i)
{

	/*prlongf("%f, ",lambda);
	if(i==worker_num-1)
	{
		prlongf("\n");
	}*/
	double lambda = calc_lambda(pLabelMat, i);
	long z = random_sample(lambda/(1+lambda));
	update_alpha_beta_y(pLabelMat, i, z);
}


void gs_burn_in(long *pLabelMat)
{
	init_y_alpha_beta(pLabelMat);
	for(long t=0; t<burn_num; ++t)
	{
		for(long i=0; i<task_num; ++i)
		{
			generate_one_label(pLabelMat, i);
		}
	}
}


double infer(long *pLabelMat)
{

	double accuracy = 0.0;
	gs_burn_in(pLabelMat);
	
	for(long i=0; i<2*task_num; ++i)
	{
		yDist[i] = 0.0;
	}

	for(long i=0; i<task_num; ++i)
	{
		xVec[i] = 0.0;
	}

	for(long j=0; j<worker_num; ++j)
	{
		pVec[j] = 0.0;
	}

	for(long i=0; i<task_num; ++i)
	{
		for(long j=0; j<worker_num;++j)
		{
			pDist[i*worker_num+j] = 0;
		}
	}

	p0 = 0;

	for(long t=0; t<sample_num; ++t)
	{
		for(long n=0; n<longerval; ++n)
		{
			for(long i=0; i<task_num; ++i)
			{
				generate_one_label(pLabelMat, i);
			}
		}

		for(long i=0; i<task_num; ++i)
		{
			yDist[2*i+y[i]]+=1.0/sample_num;
			double lambda = calc_lambda(pLabelMat, i);
			xVec[i] += log(lambda)/sample_num;
		}


		p0 += (double)beta[0]/(beta[0]+beta[1])/sample_num;

		for(long i=0; i<task_num; ++i)
		{
			for(long j=0; j<worker_num; ++j)
			{
				if(y[i]==pLabelMat[i*worker_num+j]-1)
				{
					pDist[i*worker_num+j] += 1.0/sample_num;
				}
			}
		}

		for(long j=0; j<worker_num; ++j)
		{
			pVec[j] += (double)alpha[2*j]/(alpha[2*j]+alpha[2*j+1])/sample_num;
		}
	}

	//prlongf("\n%d\t%d\t%f\n", beta[0], beta[1], p0);
	/*
	for(long i=0; i<task_num; ++i)
	{
		xVec[i] = log(p0/(1-p0));
		for(long j=0; j<worker_num; ++j)
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
	}*/

	for(long i=0; i<task_num; ++i)
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