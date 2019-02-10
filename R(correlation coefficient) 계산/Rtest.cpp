#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
int main()
{
	int i, j;
	double N;
	float avgx = 0, avgy = 0, x[500] = { 0 }, y[500] = { 0 }, max = 0, min;
	float uptemp = 0, downtemp, xtemp = 0, ytemp = 0, R;
	FILE *fp;

	fp = fopen("input.txt", "r");
	if (fp == NULL)
		printf("error\n");
	for (i = 1; fscanf(fp, "%lf", &N) == 1; ++i)
		fscanf(fp, "%f %f\n", &x[i], &y[i]);
	fclose(fp);
	i = i - 1;

	min = x[1];
	for (j = 1; j <= i; j++) //normalization x
	{
		if (x[j] > max)
			max = x[j];

		if (x[j] < min)
			min = x[j];
	}
	for (j = 1; j <= i; j++)
		x[j] = (x[j] - min) / (max - min);

	max = 0;
	min = y[1];
	for (j = 1; j <= i; j++) //normalization y
	{
		if (y[j] > max)
			max = y[j];

		if (y[j] < min)
			min = y[j];
	}
	for (j = 1; j <= i; j++)
		y[j] = (y[j] - min) / (max - min);

	fp = fopen("output.txt", "w");	// normaliztion 결과
	if (fp == NULL)
		printf("error\n");
	for (j = 1; j <= i; j++)
		fprintf(fp, "%f %f\n", x[j], y[j]);
	fclose(fp);

	for (j = 1; j <= i; j++) // R calculation
	{
		avgx += x[j];
		avgy += y[j];
	}
	avgx = avgx / (float)i;
	avgy = avgy / (float)i;
	
	for (j = 1; j <= i; j++)
		uptemp = uptemp + (x[j] - avgx)*(y[j] - avgy);
	for (j = 1; j <= i; j++)
	{
		xtemp += (x[j] - avgx)*(x[j] - avgx);
		ytemp += (y[j] - avgy)*(y[j] - avgy);
	}
	downtemp = xtemp * ytemp;
	R = uptemp / (float)(sqrt)(downtemp); // 최종 r 구하기

	printf("%f", R);

	return 0;
}