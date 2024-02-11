inlets = 1;
outlets = 1;

var imSize = 100;
var matrix = null;


function getNextPos(curX, curY, aimX, aimY) {
  var newPos = { x: curX, y: curY };

  if (isAllowed(aimX, aimY)) {
    newPos = { x: aimX, y: aimY };
    // post("inside the zone!\n")
  } else {
    // post("outside the zone\n")
  }
  if (aimX - curX > 0 && isAllowed(curX + 1, curY)) {
    newPos = { x: curX + 1, y: curY };
  } else if (aimX - curX < 0 && isAllowed(curX - 1, curY)) {
    newPos = { x: curX - 1, y: curY };
  }
  if (aimY - curY > 0 && isAllowed(curX, curY + 1)) {
    newPos = { x: curX, y: curY + 1 };
  } else if (aimY - curY < 0 && isAllowed(curX, curY - 1)) {
    newPos = { x: curX, y: curY - 1 };
  }
  return newPos;
}


function isAllowed(x, y) {
  if (x < 0 || x > imSize) {
    return false;
  }
  if (y < 0 || x > imSize) {
    return false;
  }
  // return matrix[(y*imSize + x) * 4] > 127;
//   post(y, x, "\n");

  cell = matrix.getcell(x, y);

  if (cell == null) {
    return false;
  }

  return cell[1] > 127;
}

function list() {
    // arguments : current_x current_y target_x target_y
    
	var a = arrayfromargs(arguments);

    curPos = {x: Math.floor(a[2] *imSize), y: Math.floor(a[3] *imSize) };
    aimPos = {x: Math.floor(a[0] *imSize), y: Math.floor(a[1] *imSize) };

    var numIter = 20;
    for (i=0; i<numIter; i++) {
        curPos = getNextPos(curPos.x, curPos.y, aimPos.x, aimPos.y);
    }

    outlet(0, curPos.x/imSize, curPos.y/imSize);

}


function jit_matrix(mname) {
	
	matrix = new JitterMatrix(mname);
	
}