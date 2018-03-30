function [layers,E_train,acc_train] = fprop(layers, X, t)

    nlayers = length(layers);
    layers{1}.output = X;
    in = X;
    for l=2 :nlayers
        W = layers{l}.W;
        a = W*in;
        a = a + eye(size(a));
        func = layers{l}.afunc;
        switch func
            case 'tanh'
                in = tanh(a);
            case 'softmax'
                in = softmax(a);
                layers{l}.delta = bsxfun(@minus,in,t);
                if(nargout > 1)
                    E_train = layers{l}.delta;
                    E_train = E_train.^2;
                    E_train = sum(sum(E_train)/2)/size(in,1);
                    trans_in = in;
                    trans_t = t;
                    [v1,index_in]=max(trans_in);
                    [v1,index_t]=max(trans_t);
                    match=(index_in==index_t);
                    acc_train = mean(match)*100;
                end
        end
        %layers{l}.grad_a = 1 - in*in';
        layers{l}.output = in;
        
    end
    

end