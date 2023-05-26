function result = parforprogress(TODO, lambda, message, debug)
%PARFORPROGRESS parfor with a progress bar

    if nargin<3
        message = 'Please wait ...';
    end
    if nargin<4
        debug = false; 
    end

    h      = waitbar(0, message);
    ndone  = 1;
    result = zeros(TODO,1);
    function nUpdateWaitbar(~)
        waitbar(ndone./TODO, h);
        ndone = ndone + 1;
    end

    if debug
        % Use ordinary for loop in debug mode
        for i = 1:TODO
            result(i) = feval(lambda,i);
            nUpdateWaitbar();
        end
    else
        D = parallel.pool.DataQueue;
        afterEach(D, @nUpdateWaitbar);
        parfor i = 1:TODO
            result(i) = feval(lambda,i);
            send(D, i);
        end
    end

    close(h);
end


