


#include <cstdio>
#include <iostream>
#include <random>
#include <exception>
#include <malloc.h>
#include <CL\opencl.h>
#include "setup_cl.h"
#include <FreeImage\FreeImagePlus.h>


using namespace std;


#pragma region Supporting structures to reflect vector types in OpenCL

struct float2 {

	float x, y;

	float2(const float _x, const float _y) : x(_x), y(_y) {}
};

struct float3 {

	float x, y, z;

	float3(const float _x, const float _y, const float _z) : x(_x), y(_y), z(_z) {}
};

struct float4 {

	float x, y, z, w;

	float4(const float _x, const float _y, const float _z, const float _w) : x(_x), y(_y), z(_z), w(_w) {}
};

#pragma endregion


// Custom sturct to model a voronoi region
__declspec(align(16)) struct juliaSet_region {

	// Position of region (in normalised image coordinates [0, 1]) - float4 (16 byte) alignment
	__declspec(align(16)) float2	pos;

	// Colour of region - float4 (16 byte) alignment
	__declspec(align(16)) float3	colour;
};



int main(int argc, char **argv) {

	cl_int				err = 0;
	juliaSet_region		*vRegions = nullptr;
	cl_context			context = 0;
	cl_program			program = 0;
	cl_kernel			juliaKernel = 0;
	cl_mem				regionBuffer = 0;
	cl_mem				outputImage = 0;
	cl_device_id		device = 0;
	cl_command_queue	commandQueue = 0;
	fipImage			result;

	try
	{
		//const int imageWidth = 3840 ;
		//const int imageHeight = 2160;

		const int imageWidth = 1024;
		const int imageHeight = 1024;

		//const int imageWidth = 600;
		//const int imageHeight = 600;
		const cl_int numRegions = 1000000;

		// Setup random number engine
		random_device rd;
		mt19937 mt(rd());
		auto D = uniform_real_distribution<float>(0.0f, 1.0f);

		// Create region array in host memory.  Setup seed points for voronoi graph generation.
		vRegions = (juliaSet_region*)_aligned_malloc(numRegions * sizeof(juliaSet_region), 16);
		

		if (!vRegions)
			throw exception("Cannot create Julia region array in host memory");

		for (int i = 0; i<numRegions; ++i) {

			// As noted in struct definition - store coordinates in normalised range [0, 1]
			vRegions[i].pos.x = D(mt);
			vRegions[i].pos.y = D(mt);

			// Store colour (each component in the range [0, 1] - unsigned norm (UNORM) range)
			vRegions[i].colour.x = D(mt);
			vRegions[i].colour.y = D(mt);
			vRegions[i].colour.z = D(mt);
		}



		// Create and validate the OpenCL context
		context = createContext();

		if (!context)
			throw exception("Cannot create OpenCL context");


		// Get the first device associated with the context - should be the GPU
		device = getDeviceForContext(context);

		if (!device)
			throw exception("Cannot obtain valid device ID");


		// Create the command queue
		commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, 0);

		if (!commandQueue)
			throw exception("Cannot create command queue");



		// Create the program object based on voronoi.cl
		cl_program program = createProgram(context, device, "Resources\\Kernels\\JuliaSet.cl");

		if (!program)
			throw exception("Cannot create program object");


		// Get the voronoi kernel from program object created above
		juliaKernel = clCreateKernel(program, "Julia", 0);

		if (!juliaKernel)
			throw exception("Could not create kernel");



		// Create buffer corresponding to the region array
		regionBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numRegions * sizeof(juliaSet_region), vRegions, 0);

		if (!regionBuffer)
			throw exception("Cannot create region buffer");


		// Setup output image
		cl_image_format outputFormat;
		outputFormat.image_channel_order = CL_BGRA;
		//outputFormat.image_channel_order = CL_RGBA;
		
		outputFormat.image_channel_data_type = CL_UNORM_INT8;
		
		//outputFormat.image_channel_data_type = CL_SNORM_INT8;
		//outputFormat.image_channel_data_type = CL_SIGNED_INT8;
		
		
		outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &outputFormat, imageWidth, imageHeight, 0, 0, &err);

		if (!outputImage)
			throw exception("Cannot create output image object");


		// Setup memory object -> kernel parameter bindings

		clSetKernelArg(juliaKernel, 0, sizeof(cl_mem), &regionBuffer);
		clSetKernelArg(juliaKernel, 1, sizeof(cl_mem), &outputImage);
		clSetKernelArg(juliaKernel, 2, sizeof(cl_int), &numRegions);

		// Setup worksize arrays
		size_t globalWorkSize[2] = { imageWidth, imageHeight };

		// Setup event (for profiling)
		cl_event jEvent;
	//	int numEvents = 4;
	//	cl_event eventstoWait = {};

		// Enqueue kernel
		err = clEnqueueNDRangeKernel(commandQueue, juliaKernel, 2, 0, globalWorkSize, 0, 0, 0, &jEvent);

		// Block until voronoi kernel finishes and report time taken to run the kernel
		clWaitForEvents(1, &jEvent);

		cl_ulong startTime = (cl_ulong)0;
		cl_ulong endTime = (cl_ulong)0;

		clGetEventProfilingInfo(jEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, 0);
		clGetEventProfilingInfo(jEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, 0);

		double tdelta = (double)(endTime - startTime);

		std::cout << "Time taken (in seconds) to create the Julia set = " << (tdelta * 1.0e-9) << endl;


		// Extract the resulting voronoi diagram image from OpenCL
		result = fipImage(FREE_IMAGE_TYPE::FIT_BITMAP, imageWidth, imageHeight, 32);

		if (!result.isValid())
			throw exception("Cannot create the output image");


		size_t origin[3] = { 0, 0, 0 };
		size_t region[3] = { imageWidth, imageHeight, 1 };

		err = clEnqueueReadImage(commandQueue, outputImage, CL_TRUE, origin, region, 0, 0, result.accessPixels(), 0, 0, 0);

		result.convertTo24Bits();
		BOOL saved = result.save("juliaSet.jpg");

		if (!saved)
			throw exception("Cannot save voronoi diagram");


		// Dispose of resources
		clReleaseMemObject(regionBuffer);
		clReleaseMemObject(outputImage);
		clReleaseKernel(juliaKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);

		if (vRegions)
			_aligned_free(vRegions);

		return 0;
	}
	catch (exception& err)
	{
		// Output the exception message to the console
		cout << err.what() << endl;

		// Dispose of resources
		clReleaseMemObject(regionBuffer);
		clReleaseMemObject(outputImage);
		clReleaseKernel(juliaKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
		
		if (vRegions)
			_aligned_free(vRegions);

		// Done - report error
		return 1;
	}
}
