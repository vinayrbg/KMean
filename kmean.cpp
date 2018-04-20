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
int assign_site(const float* site, float* centroids,const int k, const int d)
{
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
    int k = atoi(argv[2]);
    int d = 1;
    int num = atoi(argv[1]);
    srand(31359);
    MPI_Init(NULL, NULL);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int samp_per_proc = num/nprocs;
    if(num%nprocs > 0) //If it is not dividing equally, increasing the string size
        samp_per_proc=samp_per_proc+1;
    float* all_samp = NULL;
    float* centroids;
    float* samples;
    float* sums;
    int* counts;
    counts =malloc(k * sizeof(int));
    sums =malloc(k * sizeof(float));
    samples=malloc(samp_per_proc * sizeof(float));
    centroids =malloc(k * sizeof(float));
    int* labels;
    labels = malloc(samp_per_proc * sizeof(int));
    float* grand_sums = NULL;
    int rep;
    int* grand_counts = NULL;
    int* all_labels;
    if (rank == 0)
    {
        all_samp = build_dataSamp(samp_per_proc * nprocs);
        printf("Below are the randomaly generated training datasets:\n");
        for (int i = 0; i < num; i++)
        {
            printf("%f \t",all_samp[i]);
        }
        printf("\n");
        for (int i = 0; i < k; i++) {
            centroids[i] = all_samp[i];
        }
        print_centroids(centroids, k, d);
        grand_sums = malloc(k * d * sizeof(float));
        grand_counts = malloc(k * sizeof(int));
        all_labels = malloc(nprocs * samp_per_proc * sizeof(int));
        rep=0;
    }
    MPI_Scatter(all_samp,samp_per_proc, MPI_FLOAT, samples,d * samp_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float norm = 1.0;
    while (norm > 0.00001)
    {
        rep = rep + 1;
        MPI_Bcast(centroids, k, MPI_FLOAT,0, MPI_COMM_WORLD);
        for (int i = 0; i < k; i++){
            sums[i] = 0.0;
            counts[i] = 0;}
        float* site = samples;
        for (int i = 0; i < samp_per_proc; i++, site += d) {
            int cluster = assign_site(site, centroids, k, d);
            counts[cluster]++;
            add_site(site, &sums[cluster*d], d);
        }
        MPI_Reduce(sums, grand_sums, k, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(counts, grand_counts, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            for (int i = 0; i<k; i++) {
                for (int j = 0; j<d; j++) {
                    int dij = i + j;
                    grand_sums[dij] /= grand_counts[i];
                }
            }
            norm = distance2(grand_sums, centroids, k);
            printf("\nnorm: %f\n",norm);
            for (int i=0; i<k; i++) {
                centroids[i] = grand_sums[i];
            }
            printf("Iteration %d and new ",rep);
            print_centroids(centroids,k,d);
        }
        MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    float* site = samples;
    for (int i = 0; i < samp_per_proc; i++, site += d) {
        labels[i] = assign_site(site, centroids, k, d);
    }
    MPI_Gather(labels, samp_per_proc, MPI_INT,all_labels, samp_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    if ((rank == 0) && 1) {
        float* site = all_samp;
        printf("\nClusters: datasets and their respective centroids.\n");
        for (int i = 0;i < nprocs * samp_per_proc;i++, site += d)
        {
            printf("Centroid: %d \t", all_labels[i]);
            for (int j = 0; j < d; j++)
                printf("%f \n", site[j]);
        }
    }
    MPI_Finalize();
}






Final code:********************************
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//Building test dataset
float* build_dataSamp(const int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    for (int i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}
//To calcualte Euclidian distance between the samples
float cal_distance(const float *data1,const float *data2, int n) {
    float dist = 0.0;
    for (int i=0; i<n; i++) {
        float diff = data1[i] - data2[i];
        dist += diff * diff;
    }
    return dist;
}
//Finding the nearest centroid and assigning
int update_cluster(const float* sample, float* centroids,const int k, const int n)
{
    int best_cluster = 0;
    float best_dist = cal_distance(sample, centroids,n);
    float* centroid = centroids + n;
    for (int c = 1; c < k; c++, centroid += n) {
        float dist = cal_distance(sample, centroid,n);
        if (dist < best_dist) {
            best_cluster = c;
            best_dist = dist;
        }
    }
    return best_cluster;
}
//To add all the samples to their corresponding centroid to make seperate clusters
void add_to_cluster(const float * site, float * sum, const int d) {
    for (int i=0; i<d; i++) {
        sum[i] += site[i];
    }
}
//To display the clusters associated to particular centroid
void display(float * centroids, const int k, const int d) {
    float *p = centroids;
    printf("Centroids:\n");
    for (int i = 0; i<k; i++) {
        for (int j = 0; j<d; j++, p++) {
            printf("C%d: %f ",i, *p);
        }
        printf("\n");
    }
}
int main(int argc, char** argv)
{
    int k = atoi(argv[2]);
    int d = 1;
    int num = atoi(argv[1]);
    srand(31359);
    MPI_Init(NULL, NULL);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int samp_per_proc = num/nprocs; // Dividing samples size per process
    if(num%nprocs > 0) //If it is not dividing equally, increasing the string size
        samp_per_proc=samp_per_proc+1;
    float* all_samp = NULL;
    float* centroids;
    float* samples;
    float* sums;
    int* counts;
    counts =malloc(k * sizeof(int));
    sums =malloc(k * sizeof(float));
    samples=malloc(samp_per_proc * sizeof(float));
    centroids =malloc(k * sizeof(float));
    int* labels;
    labels = malloc(samp_per_proc * sizeof(int));
    float* grand_sums = NULL;
    int rep;
    int* grand_counts = NULL;
    int* all_labels;
    if (rank == 0)
    {
        all_samp = build_dataSamp(samp_per_proc * nprocs);
        printf("Below are the randomaly generated training datasets:\n");
        for (int i = 0; i < num; i++)
        {
            printf("%f \t",all_samp[i]);
        }
        printf("\n");
        for (int i = 0; i < k; i++) {
            centroids[i] = all_samp[i];
        }
        display(centroids, k, d);
        grand_sums = malloc(k * d * sizeof(float));
        grand_counts = malloc(k * sizeof(int));
        all_labels = malloc(nprocs * samp_per_proc * sizeof(int));
        rep=0;
    }
    // To split entire training data sets among the processes
    MPI_Scatter(all_samp,samp_per_proc, MPI_FLOAT, samples,d * samp_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float displ = 1.0;
    while (displ > 0.00001)
    {
        rep = rep + 1;
        MPI_Bcast(centroids, k, MPI_FLOAT,0, MPI_COMM_WORLD); // To pass the new centroid value among all the processes
        for (int i = 0; i < k; i++){
            sums[i] = 0.0;
            counts[i] = 0;}
        float* site = samples;
        for (int i = 0; i < samp_per_proc; i++, site += d) {
            int cluster = update_cluster(site, centroids, k, d);
            counts[cluster]++;
            add_to_cluster(site, &sums[cluster*d], d);
        }
        MPI_Reduce(sums, grand_sums, k, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);//To compute the total of distance among samples
        MPI_Reduce(counts, grand_counts, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);// To compute the number of samples persent to
        find mean
        
        if (rank == 0) {
            for (int i = 0; i<k; i++) {
                for (int j = 0; j<d; j++) {
                    int dij = i + j;
                    grand_sums[dij] /= grand_counts[i];
                }
            }
            displ = cal_distance(grand_sums, centroids, k);
            printf("\ndispl: %f  -> ",displ);
            for (int i=0; i<k; i++) {
                centroids[i] = grand_sums[i];
            }
            printf("Iteration %d and new ",rep);
            display(centroids,k,d);
        }
        MPI_Bcast(&displ, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);  // To pass the new displacement value to all the processes
    }
    
    float* site = samples;
    for (int i = 0; i < samp_per_proc; i++, site += d) {
        labels[i] = update_cluster(site, centroids, k, d);
    }
    MPI_Gather(labels, samp_per_proc, MPI_INT,all_labels, samp_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    if ((rank == 0) && 1) {
        float* site = all_samp;
        printf("\nClusters: datasets and their respective centroids.\n");
        for (int i = 0;i < nprocs * samp_per_proc;i++, site += d)
        {
            printf("Centroid:%d \t", all_labels[i]);
            for (int j = 0; j < d; j++)
                printf("%f \n", site[j]);
        }
    }
    MPI_Finalize();
}















