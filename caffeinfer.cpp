#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace nvinfer1;

static const int INPUT_H = 200;
static const int INPUT_W = 200;
static const int INPUT_CLS_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 3;
static const int DIMS = 1090;
static const int batchsize = 10;
static const float SSD_THRESH = 0.25;
static const float mean = 127.5;
static const float pixel_scale = 0.007843;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "mbox_conf_softmax";

// const std::string img_list = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/data/ssd/10image.txt";
const std::string img_list = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/data/centernet/carringBag_calibration.txt";

samplesCommon::Args gArgs;

std::vector<std::string> readTxt(std::string file)
{
  std::vector<std::string> file_vector;
  std::ifstream infile; 
  infile.open(file.data()); 
  assert(infile.is_open()); 

  std::string s;
  while(getline(infile,s))
  {
	  int nPos_1 = s.find(":");
	  file_vector.push_back(s.substr(0, nPos_1));
  }
  infile.close();
  return file_vector;
}

float* prepare_image(std::vector<string> infLst, int bs, std::vector<std::pair<int, int>>& WH){

	float *abuf = (float*)malloc(bs * INPUT_H * INPUT_W  * INPUT_CLS_SIZE * sizeof(float));

    int singleImageV = INPUT_H * INPUT_W * INPUT_CLS_SIZE;
    for(int j = 0; j < bs; j++)
    {	
        int  offset = singleImageV * j;
        std::string imageName = infLst[j];
        // std::cout << imageName << endl;
        
        cv::Mat image, im_resize, im_float, im_subtract;
        image = cv::imread(imageName);

        int imageW = image.cols;
        int imageH = image.rows;
        WH[j] = std::pair<int, int> (imageW, imageH);

        cv::resize(image, im_resize, cv::Size(INPUT_H, INPUT_W));
        im_resize.convertTo(im_float, CV_32FC3);
        im_subtract = (im_float - mean) * pixel_scale;

        std::vector<cv::Mat> input_channels(INPUT_CLS_SIZE);
        for (int i = 0; i < 3; i++) {
            input_channels[i] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, &abuf[offset + i * INPUT_H * INPUT_W]);
        }
        
        cv::split(im_subtract, input_channels);
    }
    return abuf;
}

// output the "output features"
void writeoutput(float* output, int bs){ 
    int len = bs * OUTPUT_CLS_SIZE * DIMS; 
    float *pOutput = output; 

    fstream fs;

    fs.open("output-features.txt", ios::out);

    // output:
    for (int i = 0; i < len; i++) { 
        // printf("%f", *p++);
        fs << *pOutput++;
        fs.write("\n",1);
    }

    // close the fileName
    fs.close();
    gLogInfo << "feature map saved in output.txt.";
}


void postprocess(float* output, int batchsize, std::vector<float>& umbreMax, std::vector<float>& carrinMax) {
    std::vector<std::vector<float>> umbrescores(batchsize);
    std::vector<std::vector<float>> carrinscores(batchsize);

    int offset = OUTPUT_CLS_SIZE * DIMS;
    for (int i = 0; i < batchsize; i++) {
        for (int j = 0; j < DIMS; j++) {
            umbrescores[i].push_back(output[offset * i + 3 * j + 1]);
            carrinscores[i].push_back(output[offset * i + 3 * j + 2]);
        }
        umbreMax[i] = *max_element(umbrescores[i].begin(), umbrescores[i].end());
        carrinMax[i] = *max_element(carrinscores[i].begin(), carrinscores[i].end());
        // printf("%d images:%f , %f\n", i, umbreMax[i], carrinMax[i]);
    }
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()

    int dataIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
    	outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    /* 
    Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
    Dims3 whDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME0)));
    printf("data:%d, %d, %d, %d\n", inputDims.d[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]);
    printf("wh:%d, %d, %d, %d\n", whDims.d[0], whDims.d[1], whDims.d[2], whDims.d[3]);
    */

    const int CLS_IP = INPUT_CLS_SIZE;
    const int dataSize = batchSize * CLS_IP * INPUT_H * INPUT_W;
    const int CLS_OP = OUTPUT_CLS_SIZE;
    const int commonSize = batchSize * CLS_OP * DIMS;
    const int outputSize = commonSize;

    int inputIndex{};
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], dataSize * sizeof(float))); // data

    for (int tempBuffIndex = 1; tempBuffIndex < engine.getNbBindings(); tempBuffIndex++) {
        CHECK(cudaMalloc(&buffers[tempBuffIndex], commonSize * sizeof(float))); // mbox_conf_softmax 
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);

    for (int tempBuffIndex = 1; tempBuffIndex < engine.getNbBindings(); tempBuffIndex++) {
		if (tempBuffIndex == outputIndex)
        	CHECK(cudaMemcpyAsync(output, buffers[tempBuffIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[dataIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv) {

    // load the local file engine
    // std::string engine_file = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/bin/fp16-batch10.engine";
    std::string engine_file = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/bin/int8.engine";
    std::ifstream in_file(engine_file.c_str(), std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
        fprintf(stderr, "fail to open file to write: %s\n", engine_file.c_str());
    }
    std::streampos begin, end;
    begin = in_file.tellg();
    in_file.seekg(0, std::ios::end); 
    end = in_file.tellg();
    std::size_t size = end - begin;
    fprintf(stdout, "engine file size: %lu bytes\n", size);
    in_file.seekg(0, std::ios::beg);
    std::unique_ptr<unsigned char[]> engine_data(new unsigned char[size]);
    in_file.read((char*)engine_data.get(), size);
    in_file.close(); 

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine((const void*)engine_data.get(), size, nullptr);
    assert(engine != nullptr);
    // trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // run inference
    const int outPutSize = batchsize * OUTPUT_CLS_SIZE * DIMS; 
    float output[outPutSize];
	std::vector<std::string> infLst = readTxt(img_list);
    gLogInfo << "Infering begin...";
    std::vector<float> umbreConf, carrinConf;
    for (int btNm = 0; btNm < int (infLst.size() / batchsize); btNm++){ // todo :when batch is incompelte.
        std::vector<string> fInBc(infLst.begin() + btNm * batchsize, infLst.begin() + btNm * batchsize + batchsize);
		std::vector<std::pair<int, int>> WH(batchsize); 
        std::vector<float> umbreMax(batchsize);
        std::vector<float> carrinMax(batchsize);

		float *blob = prepare_image(fInBc, batchsize, WH);
    	doInference(*context, blob, output, batchsize); 
		// writeoutput(output, batchsize);
        postprocess(output, batchsize, umbreMax, carrinMax);
        // for (int k = 0; k < batchsize; k++) printf("%dth images:%f , %f\n", k, umbreMax[k], carrinMax[k]);
		umbreConf.insert(umbreConf.end(), umbreMax.begin(), umbreMax.end());
		carrinConf.insert(carrinConf.end(), carrinMax.begin(), carrinMax.end());
        } 
    //for (int k = 0; k < umbreConf.size(); k++) printf("%dth images:%f , %f\n", k, umbreConf[k], carrinConf[k]); 

    int resLength = umbreConf.size();
    std::vector<bool> umbreRes, carrinRes;
    for (int k = 0; k < resLength; k++) {
        if (umbreConf[k] >= SSD_THRESH) umbreRes.push_back(true); else umbreRes.push_back(false);
        if (carrinConf[k] >= SSD_THRESH) carrinRes.push_back(true); else carrinRes.push_back(false);
        std::cout << k << "th images:" << umbreRes[k] << "," << carrinRes[k] << std::endl;
    }
    
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
