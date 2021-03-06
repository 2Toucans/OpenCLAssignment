#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>

class CLHandler
{
	public:
		static bool setup(cl::Platform *platform, std::vector<cl::Device> *device, cl::Context *context, int deviceType);
		static bool build(std::string kernelName, cl::Context *context, cl::Device *device, cl::Program *program);
};