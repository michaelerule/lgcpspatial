function y = blur2d(x,sigma)
    % compare to imgaussfilt()
    % we provide this in case the image processing toolbox is absent
    [W,H] = size(x);
    ix = -W/2:(W/2-1);
    iy = -H/2:(H/2-1);
    kx = exp(-(ix/sigma).^2);
    ky = exp(-(iy/sigma).^2);
    k = ky'*kx;
    k = k./sum(k,"all");
    k = fftshift(k);
    y = real(ifft2(fft2(x).*fft2(k)));
end

