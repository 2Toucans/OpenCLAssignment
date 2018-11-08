#include "CLHandler.h"

bool CLHandler::setup(cl::Platform *platform, std::vector<cl::Device> *devices, cl::Context *context, int deviceType)
{
	//Gets a list of the platforms (implementation SDKs) you have installed
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	//If a platform can't be found, setup fails
	if (platforms.size() == 0)
	{
		std::cout << "Could not find a platform." << std::endl;
		return false;
	}

	//Gets a list of all the hardware devices of a specified type that can be used for processing
	std::vector<cl::Device> myDevices;

	switch (deviceType)
	{
		case 0:
			platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &myDevices);
			break;
		case 1:
			platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &myDevices);
			break;
		case 2:
			platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &myDevices);
			break;
	}

	//If a device can't be found, setup fails
	if (myDevices.size() == 0)
	{
		std::cout << "Could not find a device." << std::endl;
		return false;
	}

	*platform = platforms[0]; //pick the first platform (for some reason mine shows 2 of the same platforms)
	*devices = myDevices; //arbitrarily pick the first device to use

	//Prints the name and version of the platform being used
	//std::cout << "Platform Name " << platform->getInfo<CL_PLATFORM_NAME>() << std::endl;
	//std::cout << "Platform Version " << platform->getInfo<CL_PLATFORM_VERSION>() << std::endl;

	//Prints the name and version of all available devices of the specified type
	for (int i = 0; i < myDevices.size(); i++)
	{
		//std::cout << "Device " << i << " Name " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		//std::cout << "Device " << i << " Version " << devices[i].getInfo<CL_DEVICE_VERSION>() << std::endl;
	}

	//Creates a context that contains the chosen device
	*context = cl::Context(myDevices);

	return true;
}

bool CLHandler::build(std::string kernelName, cl::Context *context, cl::Device *device, cl::Program *program)
{
	std::ifstream myKernelFile(kernelName);

	if (!myKernelFile.is_open())
	{
		std::cout << "Failed to open kernel file." << std::endl;
		return false;
	}

	//creates a string from the kernel files
	std::string kernelSrc(std::istreambuf_iterator<char>(myKernelFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(kernelSrc.c_str(), kernelSrc.length() + 1));

	*program = cl::Program(*context, sources);

	//prints out any errors in the kernel code
	if (program->build() != CL_SUCCESS)
	{
		std::cout << "Error building: " << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << "\n";
		return false;
	}

	return true;
}