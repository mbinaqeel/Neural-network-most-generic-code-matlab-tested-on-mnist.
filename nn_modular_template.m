%% How to set up a neural network
load mnist_uint8;

normalization_factor=max(double(train_x(:)));
train_x = double(train_x')/normalization_factor;
test_x = double(test_x')/normalization_factor;
train_y = double(train_y');
test_y = double(test_y');

%% split training set into training and validation sets
num_validation_samples=10000;
val_x=train_x(:,end-num_validation_samples+1:end);
val_y=train_y(:,end-num_validation_samples+1:end);
train_x=train_x(:,1:end-num_validation_samples);
train_y=train_y(:,1:end-num_validation_samples);

%% network parameters
% I -> 50 -> 50 -> 50 -> O
[input_size,num_train_samples]=size(train_x);
[output_size]=size(train_y,1);
num_hidden_neurons=[50 50 50];
num_layers=length(num_hidden_neurons)+2;
error_func='cross_entropy';%'squared_error'

%% optimization parameters or SGD using minibatches
eta=1e-3;
minibatch_size=100;
num_iterations=50;
gradients_ok_flag=0;%1;

%%set seed for random number generator of Matlab.
%this will create the same random numbers on any machine.
%useful for debugging and comparing with each other
rng(1);
converged = 1;
%% setup the network
layers=cell(1,num_layers);
%input layer
layers{1}.type='i';
layers{1}.sz=input_size; %input layer has only 1 slice (i.e., the input image)
%hidden layers
for l=2:num_layers-1
    layers{l}.type='h';
    layers{l}.sz=num_hidden_neurons(l-1);
    disp(layers{l}.sz);
    disp(layers{l-1}.sz);
    layers{l}.W=randn(layers{l}.sz,layers{l-1}.sz);
    layers{l}.bias=randn(layers{l}.sz,1);
    layers{l}.afunc='tanh'; %'relu', 'lrelu', 'elu', 'sigmoid'
    layers{l}.eta=eta;
end
%output layer
layers{num_layers}.type='f';
layers{num_layers}.sz=output_size;
layers{num_layers}.W=randn(layers{num_layers}.sz,layers{num_layers-1}.sz);
layers{num_layers}.bias=randn(layers{num_layers}.sz,1);
layers{num_layers}.afunc='softmax'; %'sigmoid', 'linear'
layers{num_layers}.eta=eta;
layers{num_layers}.error_func=error_func;
clear num_hidden_neurons

%SGD using minibatches
E_train=zeros(num_iterations,1);
E_val=zeros(num_iterations,1);
E_test=zeros(num_iterations,1);
acc_train=zeros(num_iterations,1);
acc_val=zeros(num_iterations,1);
acc_test=zeros(num_iterations,1);
for iter=1:num_iterations
    %decay learning rate
    if mod(iter,5)==0
        for l=2:length(layers)
            layers{l}.eta=layers{l}.eta/iter;
        end
    end
    inds=randperm(num_train_samples);
    for i=1:minibatch_size:num_train_samples
        %fprintf('Iteration %d, batch %d\n',iter,ceil(i/minibatch_size));
        idx=inds(i:min(i+minibatch_size-1,num_train_samples));
        if gradients_ok_flag==0
            [gradients_ok_flag,layers]=check_gradients(layers,train_x(:,idx),train_y(:,idx));
            if gradients_ok_flag==0
                fprintf('Gradients are incorrect. Training will not start until they are fixed.\n');
                return;
            end
        end
        [layers]=fprop(layers,train_x(:,idx),train_y(:,idx));
        [layers,grad(iter)]=bprop(layers);
        layers=update_weights(layers);


    end
    %compute errors and accuracies on training, validation and test sets
    [layers,E_train(iter),acc_train(iter)]=fprop(layers,train_x,train_y);
    [layers,E_val(iter),acc_val(iter)]=fprop(layers,val_x,val_y);
    [layers,E_test(iter),acc_test(iter)]=fprop(layers,test_x,test_y);
%     plot errors and accuracies
%     subplot(311);
%     plot(1:iter,grad(1:iter),'-o',...
%                 'LineWidth',3,...
%                 'MarkerSize',10);
%     legend('Gradients', 'Location', 'NorthEast');
    subplot(211);
    plot(1:iter,[E_train(1:iter) E_val(1:iter) E_test(1:iter)],'-o',...
                'LineWidth',3,...
                'MarkerSize',10);
    legend('train', 'val', 'test', 'Location', 'NorthEast');
    subplot(212);
    plot(1:iter,[acc_train(1:iter) acc_val(1:iter) acc_test(1:iter)],'-o',...
                'LineWidth',3,...
                'MarkerSize',10);
    legend('train', 'val', 'test', 'Location', 'SouthEast');
    drawnow;
    disp(iter);
    %check for convergence
%     if converged
%         break;
%    end
end

