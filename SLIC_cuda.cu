//SLIC cuda kernel 


#include "SLIC_cuda.h"
#define MAX_DIST 300000000





//======== device local function ============

__device__ float2 operator-(const float2 & a, const float2 & b) {return make_float2(a.x-b.x, a.y-b.y);}
__device__ float3 operator-(const float3 & a, const float3 & b) {return make_float3(a.x-b.x, a.y-b.y,a.z-b.z);}
__device__ int2 operator+(const int2 & a, const int2 & b) {return make_int2(a.x+b.x, a.y+b.y);}

__device__ float computeDistance(float2 c_p_xy, float3 c_p_Lab,float areaSpx,float wc2){

    float ds2 = pow(c_p_xy.x,2)+pow(c_p_xy.y,2);
    float dc2 = pow(c_p_Lab.x,2)+pow(c_p_Lab.y,2)+pow(c_p_Lab.z,2);
    float dist = sqrt(dc2+ds2/areaSpx*wc2);

    return dist;
}

__device__ int convertIdx(int2 wg, int lc_idx,int nBloc_per_row){

    int2 relPos2D = make_int2(lc_idx%5-2,lc_idx/5-2);
    int2 glPos2D = wg+relPos2D;

    return glPos2D.y*nBloc_per_row+glPos2D.x;
}

//============ Kernel ===============

__global__ void kRgb2CIELab(uchar4* inputImg, float4* outputImg, int width, int height)
{
    int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
    int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    uchar4 nPixel=inputImg[offset];

    float _b=(float)nPixel.x/255.0;
    float _g=(float)nPixel.y/255.0;
    float _r=(float)nPixel.z/255.0;

    float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
    float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
    float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

    x/=0.950456;
    float y3=exp(log(y)/3.0);
    z/=1.088754;

    float l,a,b;

    x = x>0.008856 ? exp(log(x)/3.0) : (7.787*x+0.13793);
    y = y>0.008856 ? y3 : 7.787*y+0.13793;
    z = z>0.008856 ? z/=exp(log(z)/3.0) : (7.787*z+0.13793);

    l = y>0.008856 ? (116.0*y3-16.0) : 903.3*y;
    a=(x-y)*500.0;
    b=(y-z)*200.0;

    float4 fPixel;
    fPixel.x=l;
    fPixel.y=a;
    fPixel.z=b;
    fPixel.w = 0;

    outputImg[offset]=fPixel;
}


__global__ void k_initClusters(float4* frameLab,float* clusters,int width, int height, int nSpxPerRow, int nSpxPerCol){

    int idx_c = blockIdx.x*blockDim.x+threadIdx.x,idx_c5=idx_c*5;
    int nSpx = nSpxPerCol*nSpxPerRow;

    if(idx_c<nSpx){

        int wSpx = width/nSpxPerRow, hSpx = height/nSpxPerCol;

        int i = idx_c/nSpxPerRow;
        int j = idx_c%nSpxPerRow;

        int x = j*wSpx+wSpx/2;
        int y = i*hSpx+hSpx/2;

        clusters[idx_c5] = frameLab[y*width+x].x;
        clusters[idx_c5+1] = frameLab[y*width+x].y;
        clusters[idx_c5+2] = frameLab[y*width+x].z;
        clusters[idx_c5+3] = x;
        clusters[idx_c5+4] = y;
    }
}


__global__ void k_assignement(int width, int height,int wSpx, int hSpx,float4* frameLab, float* labels,float* clusters,float* accAtt_g,float wc2){
    int i = blockDim.y*blockIdx.y+threadIdx.y; //px idx
    int j = blockDim.x*blockIdx.x+threadIdx.x;
    int2 wgIdx = make_int2(blockIdx.x,blockIdx.y);

    __shared__ float shareClusters[125];

    float areaSpx = wSpx*hSpx;
    float distanceMin = MAX_DIST;
    float labelMin = -1;

    if(threadIdx.x<5 && threadIdx.y<5){
        int l=threadIdx.x-2;
        int k=threadIdx.y-2;
        int2 localBlockId = make_int2(blockIdx.x+l,blockIdx.y+k);
        int idShareClust5 = (threadIdx.y*5+threadIdx.x)*5;
        if(localBlockId.x>=0 && localBlockId.x< gridDim.x && localBlockId.y>=0 && localBlockId.y<gridDim.y){
            int clusterId = localBlockId.y*gridDim.x+localBlockId.x;
            int clusterId5 = clusterId*5;

            shareClusters[idShareClust5] = clusters[clusterId5];
            shareClusters[idShareClust5+1] = clusters[clusterId5+1];
            shareClusters[idShareClust5+2] = clusters[clusterId5+2];
            shareClusters[idShareClust5+3] = clusters[clusterId5+3];
            shareClusters[idShareClust5+4] = clusters[clusterId5+4];

        }else{
            //case when out ouf bound
            shareClusters[idShareClust5] = -1;
        }
    }

    __syncthreads();


//gathering 25 clusters
    if(j<width && i<height)
    {
        int ij = (i*width+j);
        float3 px_Lab = make_float3(frameLab[ij].x,frameLab[ij].y,frameLab[ij].z);
        float2 px_xy  = make_float2(j,i);


        //compare 25 centroids
        for(int cluster_idx=0; cluster_idx<25; cluster_idx++) // cluster locaux
        {
            int cluster_idx5 = cluster_idx*5;
            if(shareClusters[cluster_idx5]!=-1){
                float2 cluster_xy = make_float2(shareClusters[cluster_idx5+3],shareClusters[cluster_idx5+4]);
                float3 cluster_Lab = make_float3(shareClusters[cluster_idx5],shareClusters[cluster_idx5+1],shareClusters[cluster_idx5+2]);

                float2 px_c_xy = px_xy-cluster_xy;
                float3 px_c_Lab = px_Lab-cluster_Lab;


                if(abs(px_c_xy.x)<wSpx && abs(px_c_xy.y)<hSpx){

                    float distTmp = fminf(computeDistance(px_c_xy, px_c_Lab,areaSpx,wc2),distanceMin);

                    if(distTmp!=distanceMin){
                        distanceMin = distTmp;
                        labelMin = convertIdx(wgIdx,cluster_idx,gridDim.x);
                    }
                }
            }
        }
        labels[ij] = labelMin;



        // =============== Accumulator : simplify update step ===================
        int labelMin6 = int(labelMin*6);
        atomicAdd(&accAtt_g[labelMin6],px_Lab.x);
        atomicAdd(&accAtt_g[labelMin6+1],px_Lab.y);
        atomicAdd(&accAtt_g[labelMin6+2],px_Lab.z);
        atomicAdd(&accAtt_g[labelMin6+3],j);
        atomicAdd(&accAtt_g[labelMin6+4],i);
        atomicAdd(&accAtt_g[labelMin6+5],1); //counter
    }
}



__global__ void k_update(int nSpx,float* clusters, float* accAtt_g)
{
    int cluster_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(cluster_idx<nSpx)
    {
        uint cluster_idx6 = cluster_idx*6;
        uint cluster_idx5 = cluster_idx*5;
        int counter = accAtt_g[cluster_idx6+5];
        if(counter != 0){
            clusters[cluster_idx5] = accAtt_g[cluster_idx6]/counter;
            clusters[cluster_idx5+1] = accAtt_g[cluster_idx6+1]/counter;
            clusters[cluster_idx5+2] = accAtt_g[cluster_idx6+2]/counter;
            clusters[cluster_idx5+3] = accAtt_g[cluster_idx6+3]/counter;
            clusters[cluster_idx5+4] = accAtt_g[cluster_idx6+4]/counter;

//reset accumulator
            accAtt_g[cluster_idx6] = 0;
            accAtt_g[cluster_idx6+1] = 0;
            accAtt_g[cluster_idx6+2] = 0;
            accAtt_g[cluster_idx6+3] = 0;
            accAtt_g[cluster_idx6+4] = 0;
            accAtt_g[cluster_idx6+5] = 0;
        }
    }
}

//============== wrapper =================

__host__ void SLIC_cuda::Rgb2CIELab( uchar4* inputImg, float4* outputImg, int width, int height )
{
    dim3 threadsPerBlock(m_wSpx, m_hSpx);
    dim3 numBlocks(m_width / threadsPerBlock.x, m_height / threadsPerBlock.y);
    kRgb2CIELab<<<numBlocks,threadsPerBlock>>>(inputImg,outputImg,width,height);

}

__host__ void SLIC_cuda::InitClusters()
{
    dim3 threadsPerBlock(NMAX_THREAD);
    dim3 numBlocks(iDivUp(m_nSpx,NMAX_THREAD));
    k_initClusters<<<numBlocks,threadsPerBlock>>>(frameLab_g,clusters_g,m_width,m_height,m_width/m_wSpx,m_height/m_hSpx);
}
__host__ void SLIC_cuda::Assignement() {
    dim3 threadsPerBlock(m_wSpx, m_hSpx);
    dim3 numBlocks(m_width / threadsPerBlock.x, m_height / threadsPerBlock.y);


    float wc2 = m_wc * m_wc;
    k_assignement <<< numBlocks, threadsPerBlock >>>(m_width, m_height, m_wSpx, m_hSpx, frameLab_g, labels_g, clusters_g, accAtt_g, wc2);

}
__host__ void SLIC_cuda::Update()
{
    dim3 threadsPerBlock(NMAX_THREAD);
    dim3 numBlocks(iDivUp(m_nSpx,NMAX_THREAD));
    k_update<<<numBlocks,threadsPerBlock>>>(m_nSpx,clusters_g,accAtt_g);
}




