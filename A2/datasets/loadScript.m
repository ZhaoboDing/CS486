% loadScript.m

load trainData.txt;  % creates the matrix "trainData"
load testData.txt;   % creates the matrix "testData"
load trainLabel.txt; % creates the vector "trainLabel"
load testLabel.txt;  % creates the vector "testLabel"

% convert trainData and testData into sparse matrices
nWords = max(max(trainData(:,2)),max(testData(:,2)));
trainDataSparse = sparse(trainData(:,1),trainData(:,2),ones(size(trainData,1),1),max(trainData(:,1)),nWords);
testDataSparse = sparse(testData(:,1),testData(:,2),ones(size(testData,1),1),max(testData(:,1)),nWords);

% load words into the cell array "words"
fid = fopen('words.txt');
for lineId = 1:nWords
  words{lineId} = fgetl(fid);
end
fclose(fid);

% clean up
clear fid lineId nWords;