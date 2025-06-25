/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	/// @brief 用于计算和分配内存的工具类
	/// @tparam T 
	/// @param chunk 通用的内存块
	/// @param ptr 被操作的指针
	/// @param count 数量
	/// @param alignment 对齐的值 128
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};
	

	// @brief 计算所需的内存大小
	// @tparam T
	// @param P 点的数量
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P); //这是关键。它调用了模板参数 T（在这里是 GeometryState）的静态方法 fromChunk
		return ((size_t)size) + 128; //这里将它的地址值强制转换为一个整数 size_t，并加上128字节的额外余量（padding），然后返回。
	}
};