//SLIC cuda kernel 


#include "SLIC_cuda.h"
#define MAX_DIST 300000000
#define NNEIGH 3





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

__global__ void kRgb2CIELab(cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height)
{
    int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
    int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    int px = blockIdx.x*blockDim.x+threadIdx.x;
    int py = blockIdx.y*blockDim.y+threadIdx.y;

    if(px<width && py<height) {
        uchar4 nPixel = tex2D<uchar4>(inputImg, px, py);//inputImg[offset];

        float _b = (float) nPixel.x / 255.0;
        float _g = (float) nPixel.y / 255.0;
        float _r = (float) nPixel.z / 255.0;

        float x = _r * 0.412453 + _g * 0.357580 + _b * 0.180423;
        float y = _r * 0.212671 + _g * 0.715160 + _b * 0.072169;
        float z = _r * 0.019334 + _g * 0.119193 + _b * 0.950227;

        x /= 0.950456;
        float y3 = exp(log(y) / 3.0);
        z /= 1.088754;

        float l, a, b;

        x = x > 0.008856 ? exp(log(x) / 3.0) : (7.787 * x + 0.13793);
        y = y > 0.008856 ? y3 : 7.787 * y + 0.13793;
        z = z > 0.008856 ? z /= exp(log(z) / 3.0) : (7.787 * z + 0.13793);

        l = y > 0.008856 ? (116.0 * y3 - 16.0) : 903.3 * y;
        a = (x - y) * 500.0;
        b = (y - z) * 200.0;

        float4 fPixel;
        fPixel.x = l;
        fPixel.y = a;
        fPixel.z = b;
        fPixel.w = 0;

        surf2Dwrite(fPixel, outputImg, px*16 , py);
    }

}


__global__ void k_initClusters(cudaSurfaceObject_t frameLab,float* clusters,int width, int height, int nSpxPerRow, int nSpxPerCol){

    int idx_c = blockIdx.x*blockDim.x+threadIdx.x,idx_c5=idx_c*5;
    int nSpx = nSpxPerCol*nSpxPerRow;

    if(idx_c<nSpx){

        int wSpx = width/nSpxPerRow, hSpx = height/nSpxPerCol;

        int i = idx_c/nSpxPerRow;
        int j = idx_c%nSpxPerRow;

        int x = j*wSpx+wSpx/2;
        int y = i*hSpx+hSpx/2;

        float4 color;
        surf2Dread(&color,frameLab, x * 16, y);


        clusters[idx_c5] = color.x;
        clusters[idx_c5+1] = color.y;
        clusters[idx_c5+2] = color.z;
        clusters[idx_c5+3] = x;
        clusters[idx_c5+4] = y;
    }
}


__global__ void k_assignement(int width, int height,int wSpx, int hSpx,cudaSurfaceObject_t frameLab, cudaSurfaceObject_t labels,float* clusters,float* accAtt_g,float wc2){

    // gather NNEIGH surrounding clusters

    __shared__ float4 sharedLab[NNEIGH][NNEIGH];
    __shared__ float2 sharedXY[NNEIGH][NNEIGH];

    int nClustPerRow = width/wSpx;
    int nn2 = NNEIGH/2;


    if(threadIdx.x<NNEIGH && threadIdx.y<NNEIGH)
    {
        int id_x = threadIdx.x-nn2;
        int id_y = threadIdx.y-nn2;

        int clustLinIdx = blockIdx.x+id_y*nClustPerRow + id_x;
        if(clustLinIdx>=0 && clustLinIdx<gridDim.x)
        {
            int clustLinIdx5 = clustLinIdx*5;
            sharedLab[threadIdx.y][threadIdx.x].x = clusters[clustLinIdx5];
            sharedLab[threadIdx.y][threadIdx.x].y = clusters[clustLinIdx5+1];
            sharedLab[threadIdx.y][threadIdx.x].z = clusters[clustLinIdx5+2];

            sharedXY[threadIdx.y][threadIdx.x].x = clusters[clustLinIdx5+3];
            sharedXY[threadIdx.y][threadIdx.x].y = clusters[clustLinIdx5+4];
        }
        else
        {
            sharedLab[threadIdx.y][threadIdx.x].x = -1;
        }

    }

    __syncthreads();
    // Find nearest neighbour

    float areaSpx = wSpx*hSpx;
    float distanceMin = MAX_DIST;
    float labelMin = -1;

    int px_in_grid = blockIdx.x*blockDim.x+threadIdx.x;
    int py_in_grid = blockIdx.y*blockDim.y+threadIdx.y;

    int px = px_in_grid%width;

    if(py_in_grid<hSpx && px<width)
    {
        int py = py_in_grid+px_in_grid/width*hSpx;
        int pxpy = py*width+px;

        float4 color;
        surf2Dread(&color,frameLab,px*16,py);

        //float3 px_Lab = make_float3(frameLab[pxpy].x,frameLab[pxpy].y,frameLab[pxpy].z);
        float3 px_Lab = make_float3(color.x,color.y,color.z);

        float2 px_xy  = make_float2(px,py);

        for(int i=0; i<NNEIGH; i++)
        {
            for(int j=0; j<NNEIGH ; j++)
            {
                if(sharedLab[i][j].x!=-1)
                {
                    float2 cluster_xy = make_float2(sharedXY[i][j].x,sharedXY[i][j].y);
                    float3 cluster_Lab = make_float3(sharedLab[i][j].x,sharedLab[i][j].y,sharedLab[i][j].z);

                    float2 px_c_xy = px_xy-cluster_xy;
                    float3 px_c_Lab = px_Lab-cluster_Lab;

                    float distTmp = fminf(computeDistance(px_c_xy, px_c_Lab,areaSpx,wc2),distanceMin);

                    if(distTmp!=distanceMin){
                        distanceMin = distTmp;
                        labelMin = blockIdx.x+(i-nn2)*nClustPerRow + (j-nn2);
                    }

                }
            }
        }
        surf2Dwrite(labelMin,labels,px*4,py);

        int labelMin6 = int(labelMin*6);
        atomicAdd(&accAtt_g[labelMin6],px_Lab.x);
        atomicAdd(&accAtt_g[labelMin6+1],px_Lab.y);
        atomicAdd(&accAtt_g[labelMin6+2],px_Lab.z);
        atomicAdd(&accAtt_g[labelMin6+3],px);
        atomicAdd(&accAtt_g[labelMin6+4],py);
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

__host__ void SLIC_cuda::Rgb2CIELab(cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height )
{
    int side = 16;
    dim3 threadsPerBlock(side,side);
    dim3 numBlocks(iDivUp(m_width,side),iDivUp(m_height,side));
    kRgb2CIELab<<<numBlocks,threadsPerBlock>>>(inputImg,outputImg,width,height);

}

__host__ void SLIC_cuda::InitClusters()
{
    dim3 threadsPerBlock(NMAX_THREAD);
    dim3 numBlocks(iDivUp(m_nSpx,NMAX_THREAD));
    k_initClusters<<<numBlocks,threadsPerBlock>>>(frameLab_surf,clusters_g,m_width,m_height,m_width/m_wSpx,m_height/m_hSpx);
}
__host__ void SLIC_cuda::Assignement() {

    int hMax = NMAX_THREAD/m_hSpx;
    int nBlockPerClust = iDivUp(m_hSpx,hMax);

    dim3 blockPerGrid(m_nSpx, nBlockPerClust);
    dim3 threadPerBlock(m_wSpx,std::min(m_hSpx,hMax));

    CV_Assert(threadPerBlock.x>=3 && threadPerBlock.y>=3);

    float wc2 = m_wc * m_wc;
    k_assignement <<< blockPerGrid, threadPerBlock >>>(m_width, m_height, m_wSpx, m_hSpx, frameLab_surf, labels_surf, clusters_g, accAtt_g, wc2);

}
__host__ void SLIC_cuda::Update()
{
    dim3 threadsPerBlock(NMAX_THREAD);
    dim3 numBlocks(iDivUp(m_nSpx,NMAX_THREAD));
    k_update<<<numBlocks,threadsPerBlock>>>(m_nSpx,clusters_g,accAtt_g);
}




