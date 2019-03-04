#define di1 800
#define id1 10
/* miscellaneous scalar operations */
/* absoulte value */
float abx(float a)
{
	float x;

	if (a >= 0.) x = a;
	if (a < 0.) x = -a;
	return (x);
}
/* a^b */
float power(float a, float b)
{
	float x;

	x = b * log(a);
	x = exp(x);
	return (x);
}
/* value of the function exp(-(1/2)*x^2 */
float fr(float a)
{
	float x;

	x = exp(-0.5*a*a);
	return (x);
}
/* root mean square */
float rms(int a, float X[di1])
{
	int i;
	float x;

	x = 0;
	for (i = 1; i <= a; ++i) {
		x = x + X[i] * X[i];
	}
	x = x / a;
	x = sqrt(x);
	return (x);
}

/* vector operations */
/* Euclidean Norm */
float norm(int a, float X[di1])
{
	int i;
	float x;

	x = 0.;
	for (i = 1; i <= a; ++i) {
		x = x + X[i] * X[i];
	}
	x = sqrt(x);
	return (x);
}
/* X.Y */
float inner(int a, float X[di1], float Y[di1])
{
	int i;
	float x;

	x = 0.;
	for (i = 1; i <= a; ++i) {
		x = x + X[i] * Y[i];
	}
	return (x);
}
/* c[i][j] = a[i]*b[j] */
void outer(int a, float X[di1], float Y[di1], float Z[di1][di1])
{
	int i, j;

	for (i = 1; i <= a; ++i) {
		for (j = 1; j <= a; ++j) {
			Z[i][j] = X[i] * Y[j];
		}
	}
}

/* matrix operations */
/* Y = AX; */
void AX1(int a, int b, float A[di1][di1], float X[di1], float Y[di1])
{
	int i, j;

	for (i = 1; i <= a; ++i) {
		Y[i] = 0.;
		for (j = 1; j <= b; ++j) {
			Y[i] = Y[i] + A[i][j] * X[j];
		}
	}
}
/* Y = A'X */
void AX2(int a, int b, float A[di1][di1], float X[di1], float Y[di1])
{
	int i, j;

	for (i = 1; i <= b; ++i) {
		Y[i] = 0.;
		for (j = 1; j <= a; ++j) {
			Y[i] = Y[i] + A[j][i] * X[j];
		}
	}
}
/* B = aA */
void scale(int a, float b, float A[di1][di1], float B[di1][di1])
{
	int i, j;

	for (i = 1; i <= a; ++i) {
		for (j = 1; j <= a; ++j) {
			B[i][j] = b * A[i][j];
		}
	}
}
/* Matrix add : C = A + B */
void Madd(int a, int b, float A[di1][di1], float B[di1][di1], float C[di1][di1])
{
	int i, j;

	for (i = 1; i <= a; ++i) {
		for (j = 1; j <= b; ++j) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
}
/* Matrix sub : C = A - B */
void Msub(int a, int b, float A[di1][di1], float B[di1][di1], float C[di1][di1])
{
	int i, j;

	for (i = 1; i <= a; ++i) {
		for (j = 1; j <= b; ++j) {
			C[i][j] = A[i][j] - B[i][j];
		}
	}
}
/* C = A*B */
void Mmul(int a, int b, int c, float A[di1][di1], float B[di1][di1], float C[di1][di1])
{
	int i, j, k;
	float x;

	for (i = 1; i <= a; ++i) {
		for (j = 1; j <= c; ++j) {
			x = 0.;
			for (k = 1; k <= b; ++k) {
				x = x + A[i][k] * B[k][j];
			}
			C[i][j] = x;
		}
	}
}

/* chaotic time-series */

float LM(float a)
{
	float x;

	x = 4 * a*(1 - a);
	return (x);
}

float MG(float a, float b)
{
	float x, y, z;

	x = 10 * log(a);
	x = exp(x);
	y = 0.2*a / (1 + x);
	z = y - 0.1*b;
	return (z);
}

/* Gaussian kernel functions */

float kernx(int a, float s, float X[id1], float M[id1])
{
	int i;
	float x, y, z;

	z = 1;
	x = 0;
	for (i = 1; i <= a; ++i) {
		x = x + (X[i] - M[i])*(X[i] - M[i]);
	}
	y = -x / (2 * s*s);
	z = z * exp(y);
	return (z);
}
/* the derivative value of Gaussian Kernel Function */
float kerny(int a, float s, float X[id1], float M[id1])
{
	int i;
	float x, y, z;

	z = 1;
	x = 0;
	for (i = 1; i <= a; ++i) {
		x = x + (X[i] - M[i])*(X[i] - M[i]);
	}
	y = -x * s;
	z = z * exp(y);
	return (z);
}
#pragma once
