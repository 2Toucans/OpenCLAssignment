#define _CRT_SECURE_NO_WARNINGS
#include "CLHandler.h"
#include <iostream>
#include <fstream>
#include <array>
#include <ctime>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

cl::Platform platform;
std::vector<cl::Device> devices;
cl::Context context;
cl::Program program;
int width, height;

//Fun Fact: Basically every OpenCL function returns an error code for debugging
//You can even pass in a reference to a cl_int to constructors to check for errors!

int main()
{
	//FIRST DO IT WITH GPU
	
	//Gets the platforms and devices to be used
	if (!CLHandler::setup(&platform, &devices, &context, 0))
		std::cin.get();

	//Builds the kernel program
	if (!CLHandler::build("myKernel.cl", &context, &devices[0], &program))
		std::cin.get();

	int err = 0; //Used to get error codes from OpenCL functions

	unsigned int* imageData = (unsigned int*)stbi_load("../img.png", &width, &height, NULL, STBI_rgb_alpha);

	int numPixels = width * height;
	
	//Inbuffer won't be changed by kernel, host doesn't need it, kernel makes copy of vec
	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * numPixels, imageData, &err);
	
	//Outbuffer will contain the kernel's output, which will be read by the host
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned int) * numPixels, nullptr, &err);

	//Create the kernel
	cl::Kernel kernel(program, "myKernel", &err);

	//Check to see if the kernel was made successfully
	if (err != 0)
	{
		std::cout << "ERROR " << err << std::endl;
		std::cin.get();
	}

	//Set the kernel arguments that correspond to the ones in the kernel file
	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, width);
	err = kernel.setArg(3, height);

	clock_t startTime = clock();

	cl::CommandQueue queue(context, devices[0]);

	//Fills buffer with 3's starting from element 0
	//queue.enqueueFillBuffer(inBuf, 3, sizeof(int) * 0, sizeof(unsigned int) * numPixels);

	//enqueueTask only executes kernel once, so we need enqueueNDRangeKernel
	//Can specify localSize to say how many work items can be in each work group
	//If unspecified compiler will decide for you
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numPixels));

	//Can also use enqueueMapBuffer, then memCopy, then unMap, to speed this up
	err = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(unsigned int) * numPixels, imageData);

	//Ensures everything is finished excecuting before continuing
	cl::finish();

	clock_t endTime = clock();
	double timeElapsed = (endTime - startTime) / 1000.0;
	startTime = endTime;

	std::cout << "Time to process = " << timeElapsed << std::endl;

	stbi_write_png("../imggpu.png", width, height, STBI_rgb_alpha, imageData, width * STBI_rgb_alpha);

	endTime = clock();
	timeElapsed = (endTime - startTime) / 1000.0;

	std::cout << "Time to write = " << timeElapsed << std::endl;


	//SECOND DO IT WITH CPU

	//Gets the platforms and devices to be used
	if (!CLHandler::setup(&platform, &devices, &context, 1))
		std::cin.get();

	//Builds the kernel program
	if (!CLHandler::build("myKernel.cl", &context, &devices[0], &program))
		std::cin.get();

	imageData = (unsigned int*)stbi_load("../img.png", &width, &height, NULL, STBI_rgb_alpha);

	numPixels = width * height;

	//Inbuffer won't be changed by kernel, host doesn't need it, kernel makes copy of vec
	inBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * numPixels, imageData, &err);

	//Outbuffer will contain the kernel's output, which will be read by the host
	outBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned int) * numPixels, nullptr, &err);

	//Create the kernel
	kernel = cl::Kernel(program, "myKernel", &err);

	//Check to see if the kernel was made successfully
	if (err != 0)
	{
		std::cout << "ERROR " << err << std::endl;
		std::cin.get();
	}

	//Set the kernel arguments that correspond to the ones in the kernel file
	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, width);
	err = kernel.setArg(3, height);

	startTime = clock();

	queue = cl::CommandQueue(context, devices[0]);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numPixels));

	//Can also use enqueueMapBuffer, then memCopy, then unMap, to speed this up
	err = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(unsigned int) * numPixels, imageData);

	//Ensures everything is finished excecuting before continuing
	cl::finish();

	endTime = clock();
	timeElapsed = (endTime - startTime) / 1000.0;
	startTime = endTime;

	std::cout << "Time to process = " << timeElapsed << std::endl;

	stbi_write_png("../imgcpu.png", width, height, STBI_rgb_alpha, imageData, width * STBI_rgb_alpha);

	endTime = clock();
	timeElapsed = (endTime - startTime) / 1000.0;

	std::cout << "Time to write = " << timeElapsed << std::endl;
	
	
	//THIRD DO IT WITH GPU AND CPU

	//Gets the platforms and devices to be used
	if (!CLHandler::setup(&platform, &devices, &context, 2))
		std::cin.get();

	//Builds the kernel program
	if (!CLHandler::build("myKernel.cl", &context, &devices[0], &program))
		std::cin.get();

	imageData = (unsigned int*)stbi_load("../img.png", &width, &height, NULL, STBI_rgb_alpha);

	//Inbuffer won't be changed by kernel, host doesn't need it, kernel makes copy of vec
	inBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * numPixels, imageData, &err);

	//Outbuffer will contain the kernel's output, which will be read by the host
	outBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned int) * numPixels, nullptr, &err);

	//Create the kernel
	kernel = cl::Kernel(program, "myKernel", &err);

	//Check to see if the kernel was made successfully
	if (err != 0)
	{
		std::cout << "ERROR " << err << std::endl;
		std::cin.get();
	}

	//Set the kernel arguments that correspond to the ones in the kernel file
	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, width);
	err = kernel.setArg(3, height);

	startTime = clock();

	int half = numPixels % 2 == 0 ? 0 : 1;

	queue = cl::CommandQueue(context, devices[0]);
	cl::CommandQueue queue2 = cl::CommandQueue(context, devices[1]);

	err = queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(numPixels / 2));
	err = queue2.enqueueNDRangeKernel(kernel, numPixels / 2, cl::NDRange(numPixels / 2 + half));

	unsigned int *result = new unsigned int[numPixels];

	//Can also use enqueueMapBuffer, then memCopy, then unMap, to speed this up
	err = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(unsigned int) * (numPixels / 2), result);
	err = queue2.enqueueReadBuffer(outBuf, CL_FALSE, sizeof(unsigned int) * numPixels / 2, sizeof(unsigned int) * (numPixels / 2 + half), result + (numPixels / 2));
	
	//Ensures everything is finished excecuting before continuing
	cl::finish();

	endTime = clock();
	timeElapsed = (endTime - startTime) / 1000.0;
	startTime = endTime;

	std::cout << "Time to process = " << timeElapsed << std::endl;

	stbi_write_png("../imgboth.png", width, height, STBI_rgb_alpha, result, width * STBI_rgb_alpha);

	endTime = clock();
	timeElapsed = (endTime - startTime) / 1000.0;

	std::cout << "Time to write = " << timeElapsed << std::endl;
	

	//FINALLY DO IT SERIALLY


	std::cin.get();
}