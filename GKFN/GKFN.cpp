/* Time-Series Prediction Based on a Network with GKFs: */
/* Parameter Estimation Based on Recursive Linear Regression */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utility.h"

#define SGN(x) ((x) > 0 ? (1):(-1)) 
#define pi 3.141596
#define id 10       /* max. dimension of input space */   
#define di 800      /* max. dimension of PFUs */
#define Nm 500     /* max. number of time series data */      
#define sc 0.5      /* initial set-up coeff. of sigma */
#define er 0.9      /* error decrement rate */
#define Ip 10       /* number of iterations for parameter estimation */
#define cut 0.01

int  N, Nt, Nf, N1, N2, Np, Ns, P2, Td, Tk, Tp, St;
float ic, si, K0;
float tsi[Nm][id], tso[Nm]; /* data */
float s[di], hn[di], ek[Nm], ck[di], m[di][id], Si[di][di]; /* param  */
float o_sse[di];
float ts[Nm];

void INITIALIZE_SIGMA(int mode);
void SIGMA_HN(int iter);
void MEAN_HN(int iter);
void OUTWEIGHT_HN(int iter);

void SETTING(int N)
{
	register int i, j, k;
 
	fprintf(stderr, "time delay = ");
	scanf_s("%d", &Td);
	fprintf(stderr, "input dimension = ");
	scanf_s("%d", &Tk);
	Ns = N - Td * (Tk - 1);
	for (i = 1; i <= Tk; ++i) {
		k = Td * (i - 1);
		for (j = 1; j <= Ns; ++j) {
			k = k + 1;
			tsi[j][i] = ts[k];
		}
	}
	fprintf(stderr, "forward prediction step = ");
	scanf_s("%d", &Tp);
	Ns = Ns - Tp;
	St = Td * (Tk - 1) + Tp;
	k = St;
	for (i = 1; i <= Ns; ++i) {
		k = k + 1;
		tso[i] = ts[k];
	}
	Nt = 1087;
	Nf = 144;
	fprintf(stderr, "N = %d  Nt = %d Nf = %d\n", N, Nt, Nf);

	/* learning parameters */
	fprintf(stderr, "initial error margin = ");
	scanf_s("%f", &ic);
	fprintf(stderr, "upper-bound of standard deviation = ");
	scanf_s("%f", &si);
	K0 = 1;
	fprintf(stderr, "number of GKFs = ");
	scanf_s("%d", &N1);
	fprintf(stderr, "number of epochs for parameter estimation = ");
	scanf_s("%d", &N2);
}

void RECRUIT_FTN(void)
{
	int   a, b, i, j, k, l;
	float c, x, y, z, ec, dk, sk;
	float yd, yk, Ck;
	float a1[id], a2[id], a3[id];
	float aka[di], akb[di], pka[di], pkb[di], Bka[di], Bkb[di];
	float Psa[di][di], Psi[di][di], dA[di][di];

	x = 1000.;
	y = 0.;
	for (j = 1; j <= Nt; ++j) {
		z = tso[j];
		if (z < x) {
			x = z;
			a = j;
		}
		if (z > y) {
			y = z;
			b = j;
		}
	}
	/*
	 * assign input data
	 */
	for (j = 1; j <= Tk; ++j) {
		a1[j] = tsi[b][j];
		m[1][j] = a1[j];
		a2[j] = tsi[a][j];
		m[2][j] = a2[j];
		a3[j] = a1[j] - a2[j];
	}

	/*
	 * Find maximum standard deviation
	 */
	x = sc * norm(Tk, a3);
	/*
if (x > si)
	x = si;
	*/
	s[1] = x;
	s[2] = x;
	Psa[1][1] = 1.;
	Psi[1][1] = 1.;
	ck[1] = tso[b];
	i = 2;
	ec = ic;


	/* recruiting gkfs */
	while (i <= N1) {
		for (l = 1; l <= Nt; ++l) {
			/* input-sample */
			for (j = 1; j <= Tk; ++j) {
				a1[j] = tsi[l][j];
				yd = tso[l];
				m[i][j] = a1[j];
			}
			/* calculation of sigma */
						/*
			x = si;
						*/
			for (j = 1; j <= i - 1; ++j) {
				for (k = 1; k <= Tk; ++k) {
					a2[k] = m[j][k];
					a3[k] = a1[k] - a2[k];
				}
				y = sc * norm(Tk, a3);
				/*
if (y < x) x = y;
				*/
			}
			/*
s[i] = x;
			*/
			s[i] = y;
			/* update pka and pkb */
			for (j = 1; j <= i; ++j) {
				for (k = 1; k <= Tk; ++k) {
					a2[k] = m[j][k];
				}
				pkb[j] = kernx(Tk, x, a2, a1);
			}
			for (j = 1; j <= i; ++j) {
				for (k = 1; k <= Tk; ++k) {
					a2[k] = m[j][k];
				}
				x = s[j];
				pka[j] = kernx(Tk, x, a1, a2);
			}
			/* check error */
			yk = inner(i - 1, pka, ck);
			ek[l] = yd - yk;
			x = ek[l];
			if (abx(x) > ec) {
				/* recruitment of a PFU */
				/* update Psa */
				for (j = 1; j <= i - 1; ++j) {
					Psa[i][j] = pka[j];
					Psa[j][i] = pkb[j];
				}
				Psa[i][i] = pka[i];
				/* calculation of Psi */
				AX2(i - 1, i - 1, Psi, pka, aka);
				AX1(i - 1, i - 1, Psi, pkb, akb);
				sk = inner(i - 1, pkb, aka);
				dk = pka[i] - sk;
				outer(i - 1, akb, aka, dA);
				Ck = 1. / dk;
				scale(i - 1, Ck, dA, dA);
				Madd(i - 1, i - 1, Psi, dA, Psi);
				for (j = 1; j <= i - 1; ++j) {
					Bka[j] = -aka[j] / dk;
					Bkb[j] = -akb[j] / dk;
				}
				for (j = 1; j <= i - 1; ++j) {
					Psi[i][j] = Bka[j];
					Psi[j][i] = Bkb[j];
				}
				Psi[i][i] = Ck;
				/* calculation of ck */
				c = ek[l] / dk;
				for (j = 1; j <= i - 1; ++j)
					ck[j] -= c * akb[j];

				ck[i] = c;
				i++;
			}
			if (i > N1) break;
		}
		/* estimation of rms-error */
		z = 0.0;
		for (j = 1; j <= Nt; ++j)
			z += ek[j] * ek[j];

		z = sqrt(z / (float)Nt);
		ec *= (float)er;
	}
	Np = i - 1;
	fprintf(stderr, "\nNp = %d\n", Np);
}

void GENERAL_INVERSE(void)
{
	register int q, p, l, j, k;
	float z, sk, dk, Ck;
	float aka[di], akb[di], ds[di], dA[di][di];

	for (q = 1; q <= N2; ++q) {
		INITIALIZE_SIGMA(1);

		for (p = 1; p <= Ip; ++p) {
			for (l = 1; l <= Nt; ++l) {

				SIGMA_HN(l);
				/* update Si */
				AX1(Np, Np, Si, hn, aka);
				sk = inner(Np, hn, aka);
				dk = 1. + sk;
				Ck = 1. / dk;
				outer(Np, aka, aka, dA);
				scale(Np, Ck, dA, dA);
				Msub(Np, Np, Si, dA, Si);

				/* adjust sigma of pfn */
				AX1(Np, Np, Si, hn, ds);
				for (j = 1; j <= Np; ++j) {
					s[j] += ds[j] * ek[l];
					if (s[j] < 0)
						s[j] = -s[j];
				}
			} /* for-loop ; l */
		} /* for-loop ; p */

		/* Mean update */
		INITIALIZE_SIGMA(2);
		for (p = 1; p <= Ip; ++p) {
			for (l = 1; l <= Nt; ++l) {

				MEAN_HN(l);

				/* update Si */
				AX1(Np*Tk, Np*Tk, Si, hn, aka);
				sk = inner(Np*Tk, hn, aka);
				dk = 1. + sk;
				Ck = 1. / dk;
				outer(Np*Tk, aka, aka, dA);
				scale(Np*Tk, Ck, dA, dA);
				Msub(Np*Tk, Np*Tk, Si, dA, Si);

				/* adjust sigma of pfn */
				AX1(Np*Tk, Np*Tk, Si, hn, ds);
				for (j = 1; j <= Np; ++j)
					for (k = 1; k <= Tk; ++k)
						m[j][k] += ds[Tk*(j - 1) + k] * ek[l];
			}
		}

		/* output weight update */
		/* initialization of Si */
		INITIALIZE_SIGMA(1);
		for (p = 1; p <= Ip; ++p) {
			for (l = 1; l <= Nt; ++l) {

				OUTWEIGHT_HN(l);

				/* update Psi */
				AX1(Np, Np, Si, hn, aka);
				AX2(Np, Np, Si, hn, akb);
				dk = 1. + inner(Np, hn, aka);
				Ck = 1. / dk;
				outer(Np, aka, akb, dA);
				scale(Np, Ck, dA, dA);
				Msub(Np, Np, Si, dA, Si);

				/* update ck */
				AX1(Np, Np, Si, hn, aka);
				for (j = 1; j <= Np; ++j)
					ck[j] += ek[l] * aka[j];
			}

			/* estimation of rms-error */
			z = 0.0;
			for (j = 1; j <= Nt; ++j)
				z += ek[j] * ek[j];

			z = sqrt(z / (float)Nt);

			/* added and changed by kmjung */
			o_sse[(q - 1)*Ip + p - 1] = z;
		}
	}
}

void SIGMA_HN(int iter)
{
	register int j, k;
	float x, y, yd, yk;
	float a1[id], a2[id], a3[di];

	/* input-sample */
	for (j = 1; j <= Tk; ++j)
		a1[j] = tsi[iter][j];
	yd = tso[iter];

	/* update pka */
	for (j = 1; j <= Np; ++j) {
		for (k = 1; k <= Tk; ++k)
			a2[k] = m[j][k];
		x = s[j];
		a3[j] = kernx(Tk, x, a1, a2);
	}

	/* check error */
	yk = inner(Np, a3, ck);
	ek[iter] = yd - yk;

	/* update hn */
	for (j = 1; j <= Np; ++j) {
		x = 1. / (2.*s[j] * s[j]);
		y = 0.0;
		for (k = 1; k <= Tk; ++k)
			y += (a1[k] - m[j][k])*(a1[k] - m[j][k]);
		hn[j] = ck[j] * y*exp(-x * y) / (s[j] * s[j] * s[j]);
	}
}

void MEAN_HN(int iter)
{
	register int j, k;
	float x, y, z, yd, yk;
	float a1[id], a2[id], a3[di];

	/* input-sample */
	for (j = 1; j <= Tk; ++j)
		a1[j] = tsi[iter][j];
	yd = tso[iter];

	/* update pka */
	for (j = 1; j <= Np; ++j) {
		for (k = 1; k <= Tk; ++k)
			a2[k] = m[j][k];
		x = s[j];
		a3[j] = kernx(Tk, x, a1, a2);
	}

	/* check error */
	yk = inner(Np, a3, ck);
	ek[iter] = yd - yk;

	/* update hn */
	for (j = 1; j <= Np; ++j) {
		x = ck[j] / (s[j] * s[j]);

		y = 0.0;
		for (k = 1; k <= Tk; ++k)
			y += (a1[k] - m[j][k])*(a1[k] - m[j][k]);
		y /= ((float)2.0*s[j] * s[j]);

		z = x * exp(-y);
		for (k = 1; k <= Tk; ++k)
			hn[Tk*(j - 1) + k] = z * (a1[k] - m[j][k]);
	}
}
void OUTWEIGHT_HN(int iter)
{
	register int j, k;
	float x, yd, yk;
	float a1[id], a2[id];

	/* input-sample */
	for (j = 1; j <= Tk; ++j)
		a1[j] = tsi[iter][j];
	yd = tso[iter];

	/* update pka */
	for (j = 1; j <= Np; ++j) {
		for (k = 1; k <= Tk; ++k)
			a2[k] = m[j][k];
		x = s[j];
		hn[j] = kernx(Tk, x, a1, a2);
	}

	/* check error */
	yk = inner(Np, hn, ck);
	ek[iter] = yd - yk;
}

void INITIALIZE_SIGMA(int mode)
{
	register int j, k;

	/* initialization of Si */
	if (mode == 1) {
		for (j = 1; j <= Np; ++j)
			for (k = 1; k <= Np; ++k) {
				if (j == k)
					Si[j][k] = K0;
				else
					Si[j][k] = 0.;
			}
	}
	else if (mode == 2) {
		for (j = 1; j <= Np * Tk; ++j)
			for (k = 1; k <= Np * Tk; ++k) {
				if (j == k)
					Si[j][k] = K0;
				else
					Si[j][k] = 0.;
			}
	}
}

void TEST(void)
{
	register int i, j, k;
	float m1, m2, x, z, w, yd, yk, sum;
	float a1[id], a2[id], pka[di];

	/* evaluation on test-samples */

	/* normalization term for training samples */
	m1 = .0;
	for (i = 1; i <= Nt; ++i) {
		m1 = m1 + tso[i];
	}
	m1 = m1 / (float)Nt;
	x = .0;

	for (i = 1; i <= Nt; ++i) {
		x = x + (tso[i] - m1)*(tso[i] - m1);
	}
	m1 = sqrt(x / ((float)Nt - 1.));
	/* normalization term for test samples */
	m2 = .0;
	for (i = Nt + 1; i <= Nt + Nf; ++i) {
		m2 = m2 + tso[i];
	}
	m2 = m2 / (float)Nf;

	x = .0;
	for (i = Nt + 1; i <= Nt + Nf; ++i) {
		x = x + (tso[i] - m2)*(tso[i] - m2);
	}
	m2 = sqrt(x / ((float)Nf - 1.));

	/* calculation of normalized rms error */
	z = .0;

	printf("yd yk ek\n");
	for (i = 1; i <= Nt + Nf; ++i) {
		/* input-sample */
		for (j = 1; j <= Tk; ++j) {
			a1[j] = tsi[i][j];
			yd = tso[i];
		}
		/* update pka */
		for (j = 1; j <= Np; ++j) {
			for (k = 1; k <= Tk; ++k) {
				a2[k] = m[j][k];
			}
			x = s[j];
			pka[j] = kernx(Tk, x, a1, a2);
		}


		/* check error */
		yk = inner(Np, pka, ck);
		ek[i] = yd - yk;
		printf("%f %f %f\n", yd, yk, ek[i]);
		z = z + ek[i] * ek[i];
		if (i == Nt) {
			w = sqrt(z / (float)(Nt - 1));
			z = 0.0;
		}
	}

	z = sqrt(z / (float)(Nf - 1));

	/* write simulation results on output-files */

	printf("mean vector ..\n");
	for (i = 1; i <= Np; ++i) {
		for (j = 1; j <= Tk; ++j) {
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
	
	printf("sigma and ck \n");
	for (i = 1; i <= Np; ++i) {
		printf("%f %f\n", s[i], ck[i]);
	}

	sum = 0.0;
	for (i = 1; i <= Nt + Nf; ++i)
		sum += fabs(ek[i]);
	sum /= (float)Nt + Nf;

	printf("Mean of absolute error = %10.4f\n", sum);

	printf("Simulation Results of a Network with GKFs\n");
	printf("adaptation of sigmas, mean vectors and output weights\n");
	printf("\n");
	printf("time delay = %d\n", Td);
	printf("\n");
	printf("input dimension = %d\n", Tk);
	printf("\n");
	printf("forward prediction step = %d\n", Tp);
	printf("\n");
	printf("number of total time series data = %d\n", N);
	printf("\n");
	printf("number of training data = %d\n", Nt);
	printf("\n");
	printf("number of test data = %d\n", Nf);
	printf("\n");
	printf("initial error margin = %f\n", ic);
	printf("\n");
	printf("upper-bound of standard deviation = %f\n", si);
	printf("\n");

	/*
printf ( "initial value of diagonal element in Si = %f\n", K0);
printf ( "\n");*/
	printf("number of epochs for recruiting GKFs = %d\n", N1);
	printf("\n");
	printf("number of epochs for parameter estimation = %d\n", N2);
	printf("\n");
	printf("number of potential functions = %d\n", Np);
	printf("\n");
	/*
printf ( "quality of inversion: %f %f\n", q0, q1);
printf ( "\n");*/
	printf("rms error for trainig-samples = %f\n", w);
	printf("\n");
	printf("standard deviation of trainig-samples = %f\n", m1);
	printf("\n");
	x = (w / m1)*100.;
	printf("normlized rms error for trainig-samples = %f percent\n", x);
	printf("\n");
	printf("rms error for test-samples = %f\n", z);
	printf("\n");
	printf("standard deviation of test-samples = %f\n", m2);
	printf("\n");
	x = (z / m2)*100.;
	printf("normlized rms error for test-samples = %f percent\n", x);
	printf("\n");
}
int main()
{
	int i;
	double X;
	FILE *fp;

	fp = fopen("ts-data.txt", "r");
	if (fp == NULL)
		printf("error\n");
	for (i = 1; fscanf_s(fp, "%lf", &X) == 1; ++i)
		fscanf_s(fp, "%f\n", &ts[i]);
	fclose(fp);

	N = i - 1;

	SETTING(N);
	RECRUIT_FTN(); /* initial set-up of PFN */
	GENERAL_INVERSE();
	TEST();

	return 0;
}