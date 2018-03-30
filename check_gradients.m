function [flag, layers] = check_gradients(layers,train_x,train_y)

    [layers,E_train]=fprop(layers,train_x,train_y);
    [layers]=bprop(layers);
    h = 0.03;
    nlayers = length(layers);
    for l = 2:nlayers
        [ith jth] = size(layers{l}.W);
        for i=1:ith
            for j=1:jth
                tempLayer = layers;
                tempLayer{l}.W(i, j) = tempLayer{l}.W(i, j)+ h;
                [~, err] = fprop(tempLayer,train_x,train_y);

                tempLayer = layers;
                tempLayer{l}.W(i, j) = tempLayer{l}.W(i, j)- h;
                [~, err2] = fprop(tempLayer,train_x,train_y);
                
                
                Ew = (err-err2)/(2*h);
                disp(err);
                disp(err2);
                disp(err-err2);
                if((abs(Ew - layers{l}.grad_e(i,j))) < h)
                    flag = 1;
                    break;
                end
            end
        end
    end
end