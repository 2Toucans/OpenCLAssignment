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
cl::Device device;
cl::Context context;
cl::Program program;
int width, height;

//Fun Fact: Basically every OpenCL function returns an error code for debugging
//You can even pass in a reference to a cl_int to constructors to check for errors!

int main()
{
	//Gets the platforms and devices to be used
	if (!CLHandler::setup(&platform, &device, &context))
		std::cin.get();

	//Builds the kernel program
	if (!CLHandler::build("myKernel.cl", &context, &device, &program))
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

	cl::CommandQueue queue(context, device);

	clock_t startTime = clock();

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

	std::cout << "Time to process = " << timeElapsed << std::endl;

	stbi_write_png("../newimg.png", width, height, STBI_rgb_alpha, imageData, width * STBI_rgb_alpha);

	timeElapsed = (clock() - endTime) / 1000.0;
	std::cout << "Time to write = " << timeElapsed << std::endl;

	std::cin.get();
}