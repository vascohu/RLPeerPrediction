#include "InferFunctions.h"
#include "SFMT\SFMT.h"
#include <stdlib.h>
#include <math.h>
#define worker_num_t (worker_num+1)


int task_num;
int worker_num;
int class_num;
int true_label_num;

int sample_num = 200;
int burn_num = 100;
int interval = 5;

sfmt_t sfmt;

int *y;
double *alpha;
double *beta;

double *yDist;
double *cmat;
double *belief;


void init_class(int _task_num, int _worker_num, int _class_num, int _true_label_num,
	int *_y, double *_alpha, double *_beta, double *_yDist, double *_cmat)
{
	task_num = _task_num;
	worker_num = _worker_num;
	class_num = _class_num;
	true_label_num = _true_label_num;

	sfmt_init_gen_rand(&sfmt, 1234);

	y = _y;
	alpha = _alpha;
	beta = _beta;
	yDist = _yDist;
	cmat = _cmat;
}

int random_sample(int start, int dim, double *p)
{
	double r = sfmt_genrand_res53(&sfmt);
	int i = 0;
	while(i<dim)
	{
		if(r<=p[i])
		{
			break;
		}
		else
		{
			r -= p[i];
			++i;
		}
	}
	return(i+start);
}

void calc_prior_dist(int *pLabelMat, int *pTrueLabel)
{
	for(int j=0; j < worker_num; ++j)
	{
		for(int k=0; k<class_num; ++k)
		{
			for(int g=0; g<class_num; ++g)
			{
				if(k==g)
				{
					alpha[j*class_num*class_num+k*class_num+g] = 2.0;
				}
				else
				{
					alpha[j*class_num*class_num+k*class_num+g] = 1.0;
				}
			}
		}
	}

	for(int k=0; k < class_num; ++k)
	{
		beta[k] = 1.0;
	}


	for(int i=0; i < true_label_num; ++i)
	{
		int k = pTrueLabel[i] - 1;
		for(int j=0; j<worker_num; ++j)
		{
			int g = pLabelMat[i*worker_num_t+j] - 1;
			if(g>=0)
			{
				alpha[j*class_num*class_num+k*class_num+g]+=1.0;
			}
		}
		beta[k]+=1.0;
	}
}


void init_y_alpha_beta(int *pLabelMat)
{
	for(int i= true_label_num; i<task_num; ++i)
	{
		double *votes = (double *) malloc(class_num * sizeof(double));
		for(int k=0; k<class_num; ++k)
		{
			votes[k] = 0.0;
		}
		for(int j=i*worker_num_t; j<(i+1)*worker_num_t-1; ++j)
		{
			int label = pLabelMat[j]-1;
			if(label>=0)
			{
				votes[label]+=1.0;
			}
		}
		double sum_votes = 0;
		for(int k=0; k<class_num; ++k)
		{
			sum_votes += votes[k];
		}
		for(int k=0; k<class_num; ++k)
		{
			votes[k] /= sum_votes;
		}
		int k = random_sample(0, class_num, votes);
		y[i] = k;
		beta[k]+=1.0;
		for(int j=0; j<worker_num; ++j)
		{
			int g = pLabelMat[i*worker_num_t+j]-1;
			if(g>=0)
			{
				alpha[j*class_num*class_num+k*class_num+g]+=1.0;
			}
		}
	}
}

void update_alpha_beta_y(int *pLabelMat, int i, int yn)
{
	int y0 = y[i];
	if(y0!=yn)
	{
		beta[y0]-=1.0;
		beta[yn]+=1.0;
		for(int j=0; j<worker_num; ++j)
		{
			int g = pLabelMat[i*worker_num_t+j]-1;
			if(g>=0)
			{
				alpha[j*class_num*class_num+y0*class_num+g]-=1.0;
				alpha[j*class_num*class_num+yn*class_num+g]+=1.0;
			}
		}
		y[i] = yn;
	}
}

double log_m_beta(double *p, int dim)
{
	double log_prob = 0.0;
	double sum_x = 0.0;
	for(int i=0; i<dim; ++i)
	{
		log_prob += lgamma(p[i]);
		sum_x += p[i];
	}
	log_prob -= lgamma(sum_x);
	return(log_prob);
}

void generate_one_label(int *pLabelMat, int i)
{
	double *log_prob = (double *) malloc(class_num* sizeof(double));
	double *prob = (double *) malloc(class_num* sizeof(double));
	for(int k=0; k<class_num; ++k)
	{
		update_alpha_beta_y(pLabelMat, i, k);
		double log_p = log_m_beta(beta, class_num);
		for(int j=0; j<worker_num; ++j)
		{
			for(int g=0; g<class_num; ++g)
			{
				log_p += log_m_beta(&alpha[j*class_num*class_num+g*class_num], class_num);
			}
		}
		log_prob[k] = log_p;
	}
	double max_log_prob = log_prob[0];
	for(int k=1; k<class_num; ++k)
	{
		if(log_prob[k]>max_log_prob)
		{
			max_log_prob = log_prob[k];
		}
	}
	double prob_sum = 0;
	for(int k=0; k<class_num; ++k)
	{
		log_prob[k] -= max_log_prob;
		prob[k] = exp(log_prob[k]);
		prob_sum += prob[k];
	}
	for(int k=0; k<class_num; ++k)
	{
		prob[k] /= prob_sum;
	}
	int z = random_sample(0, class_num, prob);
	update_alpha_beta_y(pLabelMat, i, z);
	delete(log_prob);
	delete(prob);
}


void gs_burn_in(int *pLabelMat)
{
	init_y_alpha_beta(pLabelMat);
	for(int t=0; t<burn_num; ++t)
	{
		for(int i=true_label_num; i<task_num; ++i)
		{
			generate_one_label(pLabelMat, i);
		}
	}
}


void infer(int *pLabelMat, int *true_label)
{
	calc_prior_dist(pLabelMat, true_label);
	gs_burn_in(pLabelMat);
	
	for(int i=0; i<(task_num-true_label_num)*class_num; ++i)
	{
		yDist[i] = 0.0;
	}
	for(int j=0; j<worker_num*class_num*class_num; ++j)
	{
		cmat[j] = 0.0;
	}
	for(int t=0; t<sample_num; ++t)
	{
		for(int n=0; n<interval; ++n)
		{
			for(int i=true_label_num; i<task_num; ++i)
			{
				generate_one_label(pLabelMat, i);
			}
		}
		for(int i=true_label_num; i<task_num; ++i)
		{
			yDist[(i-true_label_num)*class_num+y[i]]+=1.0/sample_num;
		}
		for(int j=0; j<worker_num*class_num*class_num; ++j)
		{
			cmat[j] += alpha[j]/sample_num;
		}
	}
}