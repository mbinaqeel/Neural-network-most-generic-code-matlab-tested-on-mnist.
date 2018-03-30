function [layers] = update_weights(layers)
    len = length(layers);
    for l=2:len
        gradient_e = layers{l}.grad_e;
        eta = layers{l}.eta;
        w = layers{l}.W;
        layers{l}.W = w - eta.*gradient_e;
    end
end
