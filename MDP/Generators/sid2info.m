function [spd, myx, y1, y2, y3] = sid2info(sid, nS, nL, nG)

tid = sid - 1;
y3  = mod(tid, nG) + 1;
tid = (tid - y3 + 1)/nG;
y2  = mod(tid, nG) + 1;
tid = (tid - y2 + 1)/nG;
y1  = mod(tid, nG) + 1;
tid = (tid - y1 + 1)/nG;
myx = mod(tid, nL) + 1;
tid = (tid - myx + 1)/nL;
spd = mod(tid, nS) + 1;

end