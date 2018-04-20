//
//  main.cpp
//  Kmean
//
//  Created by Vinay Raju on 12/2/17.
//  Copyright © 2017 Vinay Raju. All rights reserved.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
float* build_dataSamp(const int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    for (int i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}
float distance2(const float *v1, const float *v2, const int d) {
    float dist = 0.0;
    for (int i=0; i<d; i++) {
        float diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return dist;
}
int assign_site(const float* site, float* centroids,
                const int k, const int d) {
    int best_cluster = 0;
    float best_dist = distance2(site, centroids, d);
    float* centroid = centroids + d;
    for (int c = 1; c < k; c++, centroid += d) {
        float dist = distance2(site, centroid, d);
        if (dist < best_dist) {
            best_cluster = c;
            best_dist = dist;
        }
    }
    return best_cluster;
}

void add_site(const float * site, float * sum, const int d) {
    for (int i=0; i<d; i++) {
        sum[i] += site[i];
    }
}

void print_centroids(float * centroids, const int k, const int d) {
    float *p = centroids;
    printf("Centroids:\n");
    for (int i = 0; i<k; i++) {
        for (int j = 0; j<d; j++, p++) {
            printf("%f ", *p);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    int samp_per_proc = atoi(argv[3]);
    int k = atoi(argv[1]);
    int d = atoi(argv[2]);
    samp_per_proc = atoi(argv[2]);
    MPI_Init(NULL, NULL);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    float* all_samp = NULL;
    float* centroids;
    float* samples;
    float* sums;
    int* counts;
    counts =malloc(k * sizeof(int));
    sums =malloc(k * d * sizeof(float));
    samples=malloc(samp_per_proc *d* sizeof(float));
    centroids =malloc(k * sizeof(float));
    int* labels;
    labels = malloc(sites_per_proc * sizeof(int));
    float* all_sites = NULL;
    float* grand_sums = NULL;
    int* grand_counts = NULL;
    int* all_labels;
    if (rank == 0)
    {
        all_samp = build_dataSamp(samp_per_proc * nprocs);
        for (int i = 0; i < k; i++) {
            centroids[i] = all_samp[i];
        }
        print_centroids(centroids, k, d);
        grand_sums = malloc(k * d * sizeof(float));
        grand_counts = malloc(k * sizeof(int));
        all_labels = malloc(nprocs * sites_per_proc * sizeof(int));
        
    }
    MPI_Scatter(all_sites,d*sites_per_proc, MPI_FLOAT, sites,d * sites_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float norm = 1.0;
    while (norm > 0.00001)
    {
        MPI_Bcast(centroids, k*d, MPI_FLOAT,0, MPI_COMM_WORLD);
        for (int i = 0; i < k*d; i++) sums[i] = 0.0;
        for (int i = 0; i < k; i++) counts[i] = 0;
        float* site = sites;
        for (int i = 0; i < sites_per_proc; i++, site += d) {
            int cluster = assign_site(site, centroids, k, d);
            counts[cluster]++;
            add_site(site, &sums[cluster*d], d);
        }
        MPI_Reduce(sums, grand_sums, k * d, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(counts, grand_counts, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            for (int i = 0; i<k; i++) {
                for (int j = 0; j<d; j++) {
                    int dij = d*i + j;
                    grand_sums[dij] /= grand_counts[i];
                }
            }
            norm = distance2(grand_sums, centroids, d*k);
            printf("norm: %f\n",norm);
            for (int i=0; i<k*d; i++) {
                centroids[i] = grand_sums[i];
            }
            print_centroids(centroids,k,d);
        }
        MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    float* site = sites;
    for (int i = 0; i < sites_per_proc; i++, site += d) {
        labels[i] = assign_site(site, centroids, k, d);
    }
    MPI_Gather(labels, sites_per_proc, MPI_INT,
               all_labels, sites_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    if ((rank == 0) && 1) {
        float* site = all_sites;
        for (int i = 0;
             i < nprocs * sites_per_proc;
             i++, site += d) {
            for (int j = 0; j < d; j++) printf("%f ", site[j]);
            printf("%4d\n", all_labels[i]);
        }
    }
    MPI_Finalize();
}

