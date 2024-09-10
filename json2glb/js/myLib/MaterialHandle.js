function MaterialHandle() {
    this.meshArr;
    this.canvass;
    this.names;
    this.mapsIndex;
    this.fileName;
    this.resourceManager;
}
MaterialHandle.prototype={
    init:function (resourceManager,fileName) {
        this.resourceManager=resourceManager;
        this.meshArr=resourceManager.meshs;
        this.canvass=[];
        this.names=[];
        this.mapsIndex=[];
        this.fileName=fileName;
    },
    process:function () {
        for(i=0;i<this.meshArr.length;i++){
            var map1=this.meshArr[i].material.emissiveMap;
            var map2=this.meshArr[i].material.map;
            var map=map1?map1:map2;
            /*this.meshArr[i].traverse(node => {
                if (node instanceof THREE.CanvasTexture) {
                    map=node;
                    console.log(map)
                }
            });*/
            if(map){//有纹理贴图
                this.mapsIndex.push(1);
                this.names.push(this.fileName+i+'.jpg');
                var canvas=new PicHandle().getCanvas( map.image);
                this.canvass.push(canvas);
                this.meshArr[i].material=new THREE.MeshStandardMaterial({//反光材质
                    color:this.getColor(canvas)
                });
            }else this.mapsIndex.push(0);
        }
        this.resourceManager.mapsInit(this.mapsIndex,this.canvass);
    },
    getColor:function (canvas) {//取100个点求平均值
        var ctx=canvas.getContext( "2d" );  //设置画布类型2d
        var r=0,g=0,b=0,n=0;
        var dy=canvas.height<20?canvas.height:Math.floor(canvas.height/10);
        var dx=canvas.width<20?canvas.width:Math.floor(canvas.width/10);
        for( var y = 0; y < canvas.height; y+=dy )
            for( var x = 0; x < canvas.width ; x+=dx ) {
                // 获取当前位置的元素
                var pixel = ctx.getImageData( x, y, 1, 1 );//获取一个像素点的数据
                r+=pixel.data[0];
                g+=pixel.data[1];
                b+=pixel.data[2];
                n++;
            }
        //n=canvas.height*canvas.width;
        r=Math.floor(r/n);
        g=Math.floor(g/n);
        b=Math.floor(b/n);
        var color=(r*256*256+g*256+b);//.toString(16);
        console.log(color.toString(16));
        return color;
    },
    getColor0:function (canvas) {
        var ctx=canvas.getContext( "2d" );  //设置画布类型2d
        var r=0,g=0,b=0,n=0;
        for( var y = 0; y < canvas.height; y++ )
            for( var x = 0; x < canvas.width ; x++ ) {
                // 获取当前位置的元素
                var pixel = ctx.getImageData( x, y, 1, 1 );//获取一个像素点的数据
                r+=pixel.data[0];
                g+=pixel.data[1];
                b+=pixel.data[2];
            }
        n=canvas.height*canvas.width;
        r=Math.floor(r/n);
        g=Math.floor(g/n);
        b=Math.floor(b/n);
        var color=(r*256*256+g*256+b);//.toString(16);
        console.log(color.toString(16));
        return color;
    },
}

function PicHandle() {//只服务于MaterialHandle对象
    this.image;
    this.h;
    this.w;
    this.compressionRatio=0.1;//0-1
}
PicHandle.prototype={
    getCanvas:function (image) {
        var flipY = true;
        var canvas =  document.createElement( 'canvas' );
        //计算画布的宽、高
        canvas.width = image.width*this.compressionRatio;
        canvas.height = image.height*this.compressionRatio;

        var ctx = canvas.getContext( '2d' );

        if ( flipY === true ) {
            ctx.translate( 0, canvas.height );
            ctx.scale( 1, - 1 );
        }
        //将image画到画布上
        ctx.drawImage( image, 0, 0, canvas.width, canvas.height );


        /*var ctx=this.canvas.getContext( "2d" );  //设置画布类型2d
        ctx.fillStyle = "#FFFFFF";//白色
        for( var y = 0; y < this.canvas.height; y++ ) {
            for( var x = 0; x < this.canvas.width ; x++ ) {
                // 获取当前位置的元素
                var pixel = ctx.getImageData( x, y, 1, 1 );//获取一个像素点的数据
                // 判断透明度不为0
                if( pixel.data[0] +pixel.data[1] +pixel.data[2] >100) {//如果颜色较亮
                    ctx.fillRect(x,y,1,1);
                }
            }
        }*/
        return canvas;
    },

}
