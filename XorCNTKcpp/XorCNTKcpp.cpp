// XorCNTKcpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <random>
#include "CNTKLibrary.h"
#include "CNTKLibraryC.h"
#include <stdio.h>

using namespace CNTK;

inline FunctionPtr FullyConnectedLinearLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"", unsigned long seed = 1)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto timesParam = Parameter({ outputDim, inputDim }, DataType::Float, GlorotUniformInitializer(DefaultParamInitScale,
		SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, seed), device, L"timesParam");
	auto timesFunction = Times(timesParam, input, L"times");

	auto plusParam = Parameter({ outputDim }, 0.0f, device, L"plusParam");
	return Plus(plusParam, timesFunction, outputName);
}

inline FunctionPtr CreateModel(Variable input, size_t hiddenLayers, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"", unsigned long seed = 1)
{
	auto dense1 = FullyConnectedLinearLayer(input, hiddenLayers, device, L"inputLayer", seed);
	auto tanhActivation = Tanh(dense1, L"hiddenLayer");
	auto dense2 = FullyConnectedLinearLayer(tanhActivation, outputDim, device, L"outputLayer", seed);
	auto model = Sigmoid(dense2, outputName);

	return model;
}

inline TrainerPtr CreateModelTrainer(FunctionPtr model, Variable input, Variable label)
{
	auto trainingLoss = BinaryCrossEntropy(Variable(model), label, L"lossFunction");
	auto prediction = ReduceMean(Equal(label, Round(Variable(model))), Axis::AllAxes()); // Keras accuracy metric

	auto learningRatePerSample = TrainingParameterSchedule<double>(0.1, 1);
	auto parameterLearner = SGDLearner(model->Parameters(), learningRatePerSample);
	auto trainer = CreateTrainer(model, trainingLoss, prediction, { parameterLearner });

	return trainer;
}

inline void PrintTrainingProgress(TrainerPtr trainer, int minibatchIdx, int outputFrequencyInMinibatches)
{
	if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer->PreviousMinibatchSampleCount() != 0)
	{
		float trainLossValue = (float)trainer->PreviousMinibatchLossAverage();
		float evaluationValue = (float)trainer->PreviousMinibatchEvaluationAverage() * trainer->PreviousMinibatchSampleCount();

		char buffer[70];
		sprintf_s(buffer, 70, "Minibatch Epoch: %5d    loss = %8.6f    acc = %4.2f", minibatchIdx, trainLossValue, evaluationValue);
		std::cout << std::string(buffer) << std::endl;
	}
}

inline void TrainFromMiniBatchFile(TrainerPtr trainer, Variable input, Variable label, const DeviceDescriptor& device, int epochs = 1000, int outputFrequencyInMinibatches = 50)
{
	int i = 0;
	int epochs0 = epochs;

	const size_t inputDim = 2;
	const size_t numOutputClasses = 1;
	auto featureStreamName = L"features";
	auto labelsStreamName = L"labels";

	auto minibatchSource = TextFormatMinibatchSource(L"XORdataset.txt", { {featureStreamName, inputDim}, {labelsStreamName, numOutputClasses} }, MinibatchSource::InfinitelyRepeat, true);
	auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
	auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

	std::cout << std::endl << "Start Training File ..." << std::endl;

	while (epochs0 >= 0)
	{
		auto minibatchData = minibatchSource->GetNextMinibatch(4, device);

		trainer->TrainMinibatch({ { input, minibatchData[featureStreamInfo] }, { label, minibatchData[labelStreamInfo] } }, device);
		PrintTrainingProgress(trainer, i++, outputFrequencyInMinibatches);

		if (std::any_of(minibatchData.begin(), minibatchData.end(), [](const std::pair<StreamInformation, MinibatchData> & t) -> bool { return t.second.sweepEnd; }))
			epochs0--;
	}

	std::cout << "End Training File ..." << std::endl;
}

inline void TrainFromArray(TrainerPtr trainer, Variable input, Variable label, const DeviceDescriptor& device, int epochs = 1000, int outputFrequencyInMinibatches = 50)
{
	int i = 0;
	int epochs0 = epochs;

	std::vector<float> dataIn{ 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f };
	std::vector<float> dataOut{ 0.f, 1.f, 1.f, 0.f };

	std::unordered_map<Variable, MinibatchData> miniBatch;
	miniBatch[input] = MinibatchData(Value::CreateBatch(input.Shape(), dataIn, device, true), 4, 4, false);
	miniBatch[label] = MinibatchData(Value::CreateBatch(label.Shape(), dataOut, device, true), 4, 4, false);

	std::cout << std::endl << "Start Training Array ..." << std::endl;

	while (epochs0 >= 0)
	{
		trainer->TrainMinibatch(miniBatch, device);
		PrintTrainingProgress(trainer, i++, outputFrequencyInMinibatches);
		epochs0--;
	}

	std::cout << "End Training Array ..." << std::endl;
}

inline  void TestPrediction(FunctionPtr model, const DeviceDescriptor& device)
{
	std::cout << std::endl << "Prediction" << std::endl;

	auto inputVar = model->Arguments()[0];
	std::unordered_map<Variable, ValuePtr> inputDataMap;
	std::vector<float> dataIn{ 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f };
	auto inputVal = Value::CreateBatch(inputVar.Shape(), dataIn, device);
	inputDataMap[inputVar] = inputVal;

	auto outputVar = model->Output();
	std::unordered_map<Variable, ValuePtr> outputDataMap;
	outputDataMap[outputVar] = nullptr;

	model->Evaluate(inputDataMap, outputDataMap, device);
	auto outputVal = outputDataMap[outputVar];

	std::vector<std::vector<float>> inputData;
	std::vector<std::vector<float>> outputData;
	inputVal->CopyVariableValueTo(inputVar, inputData);
	outputVal->CopyVariableValueTo(outputVar, outputData);

	for (int k = 0; k < 4; ++k)
	{
		auto in0 = inputData[k];
		auto out0 = outputData[k];
		char buffer[50];
		sprintf_s(buffer, 50, "[%d %d] = %d ~ %8.6f", (int)(in0[0]), (int)(in0[1]), (int)round(out0[0]), out0[0]);
		std::cout << std::string(buffer) << std::endl;
	}
}

int main()
{
	std::mt19937_64 rng(0);
	rng.seed(time(0));
	auto seed = (unsigned long)rng() % 10000;

	auto device = DeviceDescriptor::GPUDevice(0);
	//auto device = DeviceDescriptor::CPUDevice();
	std::wstring ws = device.AsString();
	std::wcout << "XOR dataset CNTK!!! Device : " << ws << std::endl;

	const size_t inputDim = 2;
	const size_t hiddenLayers = 8;
	const size_t numOutputClasses = 1;

	auto input = InputVariable({ inputDim }, DataType::Float, L"features");
	auto label = InputVariable({ numOutputClasses }, DataType::Float, L"labels");

	auto MLPmodel = CreateModel(input, hiddenLayers, numOutputClasses, device, L"MLPmodel", seed);
	auto MLPtrainer = CreateModelTrainer(MLPmodel, input, label);

	TrainFromMiniBatchFile(MLPtrainer, input, label, device);
	//TrainFromArray(MLPtrainer, input, label, device);

	TestPrediction(MLPmodel, device);
}
