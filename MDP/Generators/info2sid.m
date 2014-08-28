function sid = info2sid(spd, myx, y1, y2, y3, nS, nL, nG)

sid = spd;
sid = (sid - 1)*nL + myx;
sid = (sid - 1)*nG + y1;
sid = (sid - 1)*nG + y2;
sid = (sid - 1)*nG + y3;

end