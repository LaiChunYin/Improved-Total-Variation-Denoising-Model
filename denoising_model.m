function denoising_model(varargin)
    p = inputParser
  
    addOptional(p, 'img_name', '/images/lena.png');
    addOptional(p, 'noise_level', 0.1); 
    addOptional(p, 'iter_num', 2500);
    addOptional(p, 'lambdas', [0.2]);
    addOptional(p, 'betas', [0.01, 0.05 , 0.1:0.05:0.25, 0.3:0.1:0.9, 1:0.2:2, 3, 5, 10, 100]);  % for the laplacian and combined modes only
    addOptional(p, 'step_size', 0.002);
    addOptional(p, 'epsilon', 0.0);  % to avoid the case the the gradient is a zero vector
    addOptional(p, 'sigma', 0.7);  % For the edge detector below
    addOptional(p, 'modes', {{'laplacian'}});  % can be one of the following: 'l2', 'tv', 'edge detector', 'test1', 'laplacian', 'combined'

    parse(p, varargin{:});

    args = fieldnames(p.Results)

    % unpack argument values
    for i = 1:length(args)
        eval([args{i}, ' = p.Results.', args{i}, ';']);
    end

    img = imread(img_name);
    img = mean(img, 3) / 255;
    [rows, cols] = size(img);
    figure; imshow(img); title('original clean image');

    % add some noise to img
    img = img + randn([rows, cols]) * noise_level;
    figure; imshow(img); title('Noisy image');

    % repeat the algorithm with different modes
    for j = 1:numel(modes)
        counter = 0;   % for plotting only
        counter = counter + 1;
        for lambda = lambdas 
            for beta = betas
                mode = modes{j};

                % let the noisy image img be the initial guess f
                f = img;
                % record the decay of f
                f_mag = [sum(abs(f), 'all')];

                % Edge detector
                conv = imgaussfilt(f, sigma);
                % approximate the gradient using forward difference
                grad_conv_x = circshift(conv, 1, 2) - conv;
                grad_conv_y = circshift(conv, 1, 1) - conv;
                g = 1 ./ (1 + (grad_conv_x .^2 + grad_conv_y .^2));

                % assume the image extend periodically
                g_forward_x = circshift(g, 1, 2);
                g_forward_y = circshift(g, 1, 1);
                g_backward_x = circshift(g, -1, 2);
                g_backward_y = circshift(g, -1, 1);

                for i = 1:iter_num
                    disp(['Iteration ' num2str(i)]);

                    grad = -(f - img);
                    % assume the image extend periodically. Note that in here, x is the
                    % horizontal direction (columns) and y is the verticle direction (rows)
                    f_forward_x = circshift(f, 1, 2);
                    f_forward_y = circshift(f, 1, 1);
                    f_backward_x = circshift(f, -1, 2);
                    f_backward_y = circshift(f, -1, 1);

                    % differences
                    forward_diff_x = f_forward_x - f;
                    forward_diff_y = f_forward_y - f;
                    backward_diff_x = f_backward_x - f;
                    backward_diff_y = f_backward_y - f;

                    % for the gradient magnatude
                    % backward x, forward y
                    a = circshift(f, [1, -1]) - f_backward_x;
                    % forward x, backward y
                    b = circshift(f, [-1, 1]) - f_backward_y;

                    % approximation of gradient
                    if strcmp(mode, 'tv')
                        grad = grad + lambda * (forward_diff_x) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2);
                        grad = grad + lambda * (backward_diff_x) ./ sqrt(epsilon + backward_diff_x .^2 +  a.^2);
                        grad = grad + lambda * (forward_diff_y) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2);
                        grad = grad + lambda * (backward_diff_y) ./ sqrt(epsilon + backward_diff_y .^2 +  b.^2);
                    end

                    % with edge detector
                    if strcmp(mode, 'edge detector')
                        grad = grad + lambda * g .* (forward_diff_x) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2);
                        grad = grad + lambda * g_backward_x .* (backward_diff_x) ./ sqrt(epsilon + backward_diff_x .^2 +  a.^2);
                        grad = grad + lambda * g .* (forward_diff_y) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2);
                        grad = grad + lambda * g_backward_y .* (backward_diff_y) ./ sqrt(epsilon + backward_diff_y .^2 +  b.^2);
                    end

                    if strcmp(mode, 'test1')    
                        grad = grad + lambda * (((forward_diff_x) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2)) + 4 * forward_diff_x .* (forward_diff_x.^2+forward_diff_y.^2));
                        grad = grad + lambda * (((backward_diff_x) ./ sqrt(epsilon + backward_diff_x .^2 +  a.^2)) + 4 * backward_diff_x .* (backward_diff_x.^2+a.^2));
                        grad = grad + lambda * (((forward_diff_y) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2)) + 4 * forward_diff_y .* (forward_diff_x.^2+forward_diff_y.^2));
                        grad = grad + lambda * (((backward_diff_y) ./ sqrt(epsilon + backward_diff_y .^2 +  b.^2)) + 4 * backward_diff_y .* (backward_diff_y.^2+b.^2));       

                    end

                    if strcmp(mode, 'l2')
                        grad = grad + lambda * 2*(forward_diff_x);
                        grad = grad + lambda * 2*(backward_diff_x);
                        grad = grad + lambda * 2*(forward_diff_y);
                        grad = grad + lambda * 2*(backward_diff_y);
                    end

                    % proposed model 1, change the TV norm of gradient to TV norm of laplacian
                    if strcmp(mode, 'laplacian')
                        f_forward_x2 = circshift(f, 2, 2);
                        f_forward_y2 = circshift(f, 2, 1);
                        f_backward_x2 = circshift(f, -2, 2);
                        f_backward_y2 = circshift(f, -2, 1);

                        f_forwx_forwy = circshift(f, [1 1]);
                        f_backx_backy = circshift(f, [-1 -1]);
                        f_forwx_backy = circshift(f, [-1 1]);
                        f_backx_forwy = circshift(f, [1 -1]);

                        % laplacians evaluated at points (x,y) (x-1,y), (x+1,y), (x,y-1), (x,y+1)
                        laplacian_x_y = f_forward_x + f_backward_x + f_forward_y + f_backward_y - 4 * f;
                        laplacian_backx_y = f + f_backward_x2 + f_backx_forwy + f_backx_backy - 4 * f_backward_x;
                        laplacian_forwx_y = f_forward_x2 + f + f_forwx_forwy + f_forwx_backy - 4 * f_forward_x;
                        laplacian_x_backy = f_forwx_backy + f_backx_backy + f + f_backward_y2 - 4 * f_backward_y;
                        laplacian_x_forwy = f_forwx_forwy + f_backx_forwy + f_forward_y2 + f - 4 * f_forward_y;

                        grad = grad + beta * 4 * (laplacian_x_y) ./ (epsilon + abs(laplacian_x_y));
                        grad = grad - beta * (laplacian_backx_y) ./ (epsilon + abs(laplacian_backx_y));
                        grad = grad - beta * (laplacian_forwx_y) ./ (epsilon + abs(laplacian_forwx_y));
                        grad = grad - beta * (laplacian_x_backy) ./ (epsilon + abs(laplacian_x_backy));
                        grad = grad - beta * (laplacian_x_forwy) ./ (epsilon + abs(laplacian_x_forwy));
                    end
                    
                    if strcmp(mode, 'laplacian (edge)')
                        % f(x+2, y), f(x, y+2), f(x, y+2), f(x, y-2)
                        f_forward_x2 = circshift(f, 2, 2);
                        f_forward_y2 = circshift(f, 2, 1);
                        f_backward_x2 = circshift(f, -2, 2);
                        f_backward_y2 = circshift(f, -2, 1);
                        % f(x+1, y+1), f(x-1, y-1), f(x+1, y-1), f(x-1, y+1)
                        f_forwx_forwy = circshift(f, [1 1]);
                        f_backx_backy = circshift(f, [-1 -1]);
                        f_forwx_backy = circshift(f, [-1 1]);
                        f_backx_forwy = circshift(f, [1 -1]);

                        % laplacians evaluated at points (x,y) (x-1,y), (x+1,y), (x,y-1), (x,y+1)
                        laplacian_x_y = f_forward_x + f_backward_x + f_forward_y + f_backward_y - 4 * f;
                        laplacian_backx_y = f + f_backward_x2 + f_backx_forwy + f_backx_backy - 4 * f_backward_x;
                        laplacian_forwx_y = f_forward_x2 + f + f_forwx_forwy + f_forwx_backy - 4 * f_forward_x;
                        laplacian_x_backy = f_forwx_backy + f_backx_backy + f + f_backward_y2 - 4 * f_backward_y;
                        laplacian_x_forwy = f_forwx_forwy + f_backx_forwy + f_forward_y2 + f - 4 * f_forward_y;

                        % with edge detector
                        grad = grad + 4 * beta * g .* (laplacian_x_y) ./ (epsilon + abs(laplacian_x_y));
                        grad = grad - beta * g_backward_x .* (laplacian_backx_y) ./ (epsilon + abs(laplacian_backx_y));
                        grad = grad - beta * g_forward_x .* (laplacian_forwx_y) ./ (epsilon + abs(laplacian_forwx_y));
                        grad = grad - beta * g_backward_y .* (laplacian_x_backy) ./ (epsilon + abs(laplacian_x_backy));
                        grad = grad - beta * g_forward_y .* (laplacian_x_forwy) ./ (epsilon + abs(laplacian_x_forwy));
                    end

                    % proposed model 2: combine the tv model with the laplacian
                    if strcmp(mode, 'combined')
                        % f(x+2, y), f(x, y+2), f(x, y+2), f(x, y-2)
                        f_forward_x2 = circshift(f, 2, 2);
                        f_forward_y2 = circshift(f, 2, 1);
                        f_backward_x2 = circshift(f, -2, 2);
                        f_backward_y2 = circshift(f, -2, 1);
                        % f(x+1, y+1), f(x-1, y-1), f(x+1, y-1), f(x-1, y+1)
                        f_forwx_forwy = circshift(f, [1 1]);
                        f_backx_backy = circshift(f, [-1 -1]);
                        f_forwx_backy = circshift(f, [-1 1]);
                        f_backx_forwy = circshift(f, [1 -1]);

                        % laplacians evaluated at points (x,y) (x-1,y), (x+1,y), (x,y-1), (x,y+1)
                        laplacian_x_y = f_forward_x + f_backward_x + f_forward_y + f_backward_y - 4 * f;
                        laplacian_backx_y = f + f_backward_x2 + f_backx_forwy + f_backx_backy - 4 * f_backward_x;
                        laplacian_forwx_y = f_forward_x2 + f + f_forwx_forwy + f_forwx_backy - 4 * f_forward_x;
                        laplacian_x_backy = f_forwx_backy + f_backx_backy + f + f_backward_y2 - 4 * f_backward_y;
                        laplacian_x_forwy = f_forwx_forwy + f_backx_forwy + f_forward_y2 + f - 4 * f_forward_y;

                        grad = grad + beta * 4 * (laplacian_x_y) ./ (epsilon + abs(laplacian_x_y));
                        grad = grad - beta * (laplacian_backx_y) ./ (epsilon + abs(laplacian_backx_y));
                        grad = grad - beta * (laplacian_forwx_y) ./ (epsilon + abs(laplacian_forwx_y));
                        grad = grad - beta * (laplacian_x_backy) ./ (epsilon + abs(laplacian_x_backy));
                        grad = grad - beta * (laplacian_x_forwy) ./ (epsilon + abs(laplacian_x_forwy));

                        grad = grad + lambda * (((forward_diff_x) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2)) + 4 * forward_diff_x .* (forward_diff_x.^2+forward_diff_y.^2));
                        grad = grad + lambda * (((backward_diff_x) ./ sqrt(epsilon + backward_diff_x .^2 +  a.^2)) + 4 * backward_diff_x .* (backward_diff_x.^2+a.^2));
                        grad = grad + lambda * (((forward_diff_y) ./ sqrt(epsilon + forward_diff_x .^2 + forward_diff_y .^2)) + 4 * forward_diff_y .* (forward_diff_x.^2+forward_diff_y.^2));
                        grad = grad + lambda * (((backward_diff_y) ./ sqrt(epsilon + backward_diff_y .^2 +  b.^2)) + 4 * backward_diff_y .* (backward_diff_y.^2+b.^2));       
                    end


                    % next iteration (note that the grad above is actually the negative gradient)
                    f = f + step_size * grad;

                    % record the decay of f
                    f_mag = [f_mag, sum(abs(f), 'all')];

                end

                % show the parameters in the title
                output_subtitle = ['(', img_name, ', ', mode, ', iterations = ', num2str(iter_num), ', noise = ', num2str(noise_level), ...
                            ', step size = ', num2str(step_size)];
                if strcmp(mode, 'tv')
                    output_subtitle = [output_subtitle, [', \lambda = ', num2str(lambda), ')']];
                end    
                if strcmp(mode, 'laplacian')
                    output_subtitle = [output_subtitle, [', \beta = ', num2str(beta), ')']];
                end
                if strcmp(mode, 'combined')
                    output_subtitle = [output_subtitle, [', \lambda = ', num2str(lambda)]];
                    output_subtitle = [output_subtitle, [', \beta = ', num2str(beta), ')']];
                end
                % show figure every 5 lambdas or betas
                if mod(counter, 5) == 0
                    fig = figure('visible', 'off'); imshow(f); title('Output image'); 
                    text(0.5, -0.05, output_subtitle, 'Units', 'normalized', 'HorizontalAlignment', 'center'); 
                    drawnow();
                end
                
            
                save_fig(f, [beta lambda], img_name, './testing/', 'testing', output_subtitle);     
            
            end
        end
    end
end


function save_fig(img, ith_iter, img_name, folder_name, save_name, output_subtitle)

        fig = figure('visible', 'off'); imshow(img); title('Output image'); 
        text(0.5, -0.05, output_subtitle, 'Units', 'normalized', 'HorizontalAlignment', 'center'); 
        drawnow();


        saveas(fig, [folder_name, save_name, num2str(ith_iter), '.jpg'], 'jpg');

        close(fig);

end