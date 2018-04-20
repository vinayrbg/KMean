#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//To calcualte Euclidian distance between the samples
float cal_distance(const float *data1,const float *data2, int n);
//Finding the nearest centroid and assigning
int update_cluster(const float* sample, float* centroids,const int k, const int n);
//To add all the samples to their corresponding centroid to make seperate clusters
void add_to_cluster(const float * site, float * sum, const int d);
//To display the clusters associated to particular centroid
void display(float * centroids, const int k, const int d);
//Building test dataset
float* build_dataSamp(const int num_elements);
//Main fuction
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
    float* opt_sums = NULL;
    int rep;
    int* opt_counts = NULL;
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
        opt_sums = malloc(k * d * sizeof(float));
        opt_counts = malloc(k * sizeof(int));
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
        MPI_Reduce(sums, opt_sums, k, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);//To compute the total of distance among samples
        MPI_Reduce(counts, opt_counts, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);// To compute the number of samples persent to find mean
        
        if (rank == 0) {
            for (int i = 0; i<k; i++) {
                for (int j = 0; j<d; j++) {
                    int tmp = i + j;
                    opt_sums[tmp] /= opt_counts[i];
                }
            }
            displ = cal_distance(opt_sums, centroids, k);
            printf("\ndispl: %f  -> ",displ);
            for (int i=0; i<k; i++) {
                centroids[i] = opt_sums[i];
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
//****Subfuctions*****
float* build_dataSamp(const int n) {
    float *dataset = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        dataset[i] = (rand()/ (float) rand());
    }
    return dataset;
}
void add_to_cluster(const float * sample, float * add, const int d) {
    for (int i=0; i<d; i++){
        add[i] += sample[i];
    }
}
float cal_distance(const float *data1,const float *data2, int n) {
    float dist = 0.0;
    for (int i=0; i<n; i++) {
        float diff = data1[i] - data2[i];
        dist = dist + diff * diff;
    }
    return dist;
}
void display(float * centroids, const int k, const int d) {
    float *cent = centroids;
    printf("Centroids:\n");
    for (int i = 0; i<k; i++) {
        for (int j = 0; j<d; j++, cent++) {
            printf("C%d: %f ",i, *cent);
        }
        printf("\n");
    }
}
int update_cluster(const float* sample, float* centroids,const int k, const int n)
{
    int final = 0;
    float min_dist = cal_distance(sample, centroids,n);
    float* centroid = centroids + n;
    for (int c = 1; c < k; c++, centroid += n) {
        float dist = cal_distance(sample, centroid,n);
        if (dist < min_dist) {
            final = c;
            min_dist = dist;
        }
    }
    return final;
}
/*****END******/


int displ[npes];
for(int i=0; i<npes; i++){
    displ[i]=i*sn;
    sendcnts[i]=sn;
    recvcnts[i]=sn;
}
if(myrank ==0 && npes != 4)
{
    printf("Please enter the node value as 4\n Ex: mpiexec -n 4:2 ./collect 17 \n");
    return 0;
}
// Clearing strings by assigning NULL
memset(s1,'\0',sizeof(s1));
memset(s2,'\0',sizeof(s2));
memset(s3,'\0',sizeof(s3));
memset(s5,'\0',sizeof(s5));
MPI_Scatterv(input, sendcnts,displ, MPI_CHAR, rcvbuf, sn, MPI_CHAR, 0, MPI_COMM_WORLD); //Scatter function
qsort(rcvbuf, strlen(rcvbuf), sizeof(char), compare);
printf("Input: %s  -> RANK: %d \t Sorted substring: %s\n",input,myrank,rcvbuf);
MPI_Gatherv(rcvbuf, sn, MPI_CHAR, rcv, recvcnts,displ, MPI_CHAR, 0,MPI_COMM_WORLD ); //Gather function
