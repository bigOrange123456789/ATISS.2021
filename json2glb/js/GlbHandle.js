//将glb分解为多个小文件
function GlbHandle(){
    this.index=0;
    this.fileName=name;
    this.download=new Download();
    this.myGlbSplit=new GlbSplit();
    this.myInDe=new InDe();
    this.duplication=new DuplicateRemoval();
    this.resourceInfo={}
    this.resourceManager=new ResourceManager();
    this.needCorrectName=false;
}
GlbHandle.prototype={
    process:function (opt) {
        console.log(opt);
        var name=opt.name;
        var glb=opt.glb;
        var needDownload=opt.needDownload===undefined?true:opt.glb;
        var needCorrectName=opt.needCorrectName===undefined?false:opt.needCorrectName;

        this.resourceManager.name=name;
        this.myGlbSplit.getArray(glb,this.resourceManager);//拆分、去除某些部件
        this.myInDe.process(this.resourceManager);//使用兴趣度进行排序

        //this.duplication.process(this.resourceManager);//模型去重

        var myMaterialHandle=new MaterialHandle();
        myMaterialHandle.init(this.resourceManager,this.fileName);
        myMaterialHandle.process();

        console.log(this.resourceManager);
        if(needCorrectName)this.correctName();

        this.download.jsonDownload(//下载说明信息
            this.resourceManager.resourceInfoGet(),"resourceInfo.json"
        );

        console.log(opt)
        console.log(needDownload)
        if(needDownload){

            var scope=this;
            this.downloadMap(myMaterialHandle,function () {//下载贴图
                scope.downloadModel(scope.resourceManager.meshs)//下载网格
            });//纹理和网格分开下载
            /**/
        }


    },
    correctName:function(){
        for(var i=0;i<this.resourceManager.meshs.length;i++){
            this.resourceManager.models[i].fileName=this.resourceManager.meshs[i].name;
        }
    },
    downloadMap:function (myMaterialHandle,finishFunction) {
        var scope=this;

        var myInterval=setInterval(function () {
            //下载图片
            if(myMaterialHandle.names.length===0)console.log("没有找到纹理贴图");
            else
            scope.download.canvasDownload(
                myMaterialHandle.canvass[scope.index],
                myMaterialHandle.names[scope.index]
            );
            scope.index++;
            if(scope.index>=myMaterialHandle.names.length){
                scope.index=0;
                clearInterval(myInterval);
                scope.resourceInfo.mapsIndex=myMaterialHandle.mapsIndex;
                if(finishFunction)finishFunction();
            }
        },1000);//有时不需要下载也要等待，这需要优化
    },
    downloadModel:function (arr) {
        var scope=this;
        var myInterval2=setInterval(function () {//下载模型
            var name=scope.fileName+scope.index+'.gltf';
            scope.download.meshDownload(arr[scope.index],name);
            scope.index++;
            if(scope.index===arr.length)clearInterval(myInterval2);
        },1000);
    },
    getGrid:function (glb,minY,maxY) {
        var myObstacle=[];
        this.resourceManager.name=name;
        this.myGlbSplit.getArray(glb,this.resourceManager);//拆分、去除某些部件
        var allMax;
        var allMin;
        for(var k=0;k<this.resourceManager.meshs.length;k++){
            var node=this.resourceManager.meshs[k];
            var geometry=node.geometry;
            var position=geometry.attributes.position;
            var arr=position.array;
            var matrix=node.matrix;
            console.log(matrix);
            var e=matrix.elements;
            for(var i=0;i<position.count;i++){
                var x=arr[3*i];
                var y=arr[3*i+1];
                var z=arr[3*i+2];
                arr[3*i]  =x*e[0]+y*e[4]+z*e[8]+e[12];
                arr[3*i+1]=x*e[1]+y*e[5]+z*e[9]+e[13];
                arr[3*i+2]=x*e[2]+y*e[6]+z*e[10]+e[14];
            }
            var index=geometry.index;
            arr=index.array;
            for(i=0;i<index.count/3;i++){
                var i_p1=arr[3*i];
                var i_p2=arr[3*i+1];
                var i_p3=arr[3*i+2];
                var p1=getPosByIndex(position,i_p1);
                var p2=getPosByIndex(position,i_p2);
                var p3=getPosByIndex(position,i_p3);
                var min=[p1[0],p1[1],p1[2]];
                var max=[p1[0],p1[1],p1[2]];

                if(p2[0]<min[0])min[0]=p2[0];
                if(p2[1]<min[1])min[1]=p2[1];
                if(p2[2]<min[2])min[2]=p2[2];

                if(p2[0]>max[0])max[0]=p2[0];
                if(p2[1]>max[1])max[1]=p2[1];
                if(p2[2]>max[2])max[2]=p2[2];

                if(p3[0]<min[0])min[0]=p3[0];
                if(p3[1]<min[1])min[1]=p3[1];
                if(p3[2]<min[2])min[2]=p3[2];

                if(p3[0]>max[0])max[0]=p3[0];
                if(p3[1]>max[1])max[1]=p3[1];
                if(p3[2]>max[2])max[2]=p3[2];

                if(min[1]>maxY||max[1]<minY) {
                } else {
                    for(var k1=Math.round(min[0]);k1<=Math.round(max[0]);k1++)
                        for(var k2=Math.round(min[2]);k2<=Math.round(max[2]);k2++){
                            myObstacle.push([k1,k2]);
                        }
                    if(allMax){
                        for(var ii=0;ii<3;ii++){
                            if(max[ii]>allMax[ii])allMax[ii]=max[ii];
                            if(min[ii]<allMin[ii])allMin[ii]=min[ii];
                        }
                    }else{
                        allMax=max;
                        allMin=min;
                    }
                }
                function getPosByIndex(position,i0){//通过索引获取顶点
                    return [
                        position.array[3*i0],
                        position.array[3*i0+1],
                        position.array[3*i0+2]
                    ];
                }
            }
        }
        this.download.jsonDownload(//下载说明信息
            {
                obstacle:myObstacle,
                allMax:allMax,
                allMin:allMin
            },"grid.json"
        );
        console.log(allMax,allMin);
    },
    eigenvector0:function (glb) {
        var myObstacle=[];
        this.resourceManager.name=name;
        this.myGlbSplit.getArray(glb,this.resourceManager);//拆分、去除某些部件
        var allMax;
        var allMin;
        for(var k=0;k<this.resourceManager.meshs.length;k++){
            var node=this.resourceManager.meshs[k];
            var geometry=node.geometry;
            var position=geometry.attributes.position;
            var arr=position.array;
            //console.log(node.name+"--"+position.count);
            var matrix=node.matrix;
            console.log(matrix);
            var e=matrix.elements;
            for(var i=0;i<position.count;i++){
                var x=arr[3*i];
                var y=arr[3*i+1];
                var z=arr[3*i+2];
                arr[3*i]  =x*e[0]+y*e[4]+z*e[8]+e[12];
                arr[3*i+1]=x*e[1]+y*e[5]+z*e[9]+e[13];
                arr[3*i+2]=x*e[2]+y*e[6]+z*e[10]+e[14];
            }
            var index=geometry.index;
            arr=index.array;
            for(i=0;i<index.count/3;i++){
                var i_p1=arr[3*i];
                var i_p2=arr[3*i+1];
                var i_p3=arr[3*i+2];
                var p1=getPosByIndex(position,i_p1);
                var p2=getPosByIndex(position,i_p2);
                var p3=getPosByIndex(position,i_p3);
                var min=[p1[0],p1[1],p1[2]];
                var max=[p1[0],p1[1],p1[2]];
                if(p2[0]<min[0])min[0]=p2[0];
                if(p2[1]<min[1])min[1]=p2[1];
                if(p2[2]<min[2])min[2]=p2[2];

                if(p2[0]>max[0])max[0]=p2[0];
                if(p2[1]>max[1])max[1]=p2[1];
                if(p2[2]>max[2])max[2]=p2[2];

                if(p3[0]<min[0])min[0]=p3[0];
                if(p3[1]<min[1])min[1]=p3[1];
                if(p3[2]<min[2])min[2]=p3[2];

                if(p3[0]>max[0])max[0]=p3[0];
                if(p3[1]>max[1])max[1]=p3[1];
                if(p3[2]>max[2])max[2]=p3[2];

                for(var k1=Math.round(min[0]);k1<=Math.round(max[0]);k1++)
                    for(var k2=Math.round(min[1]);k2<=Math.round(max[1]);k2++)
                        for(var k3=Math.round(min[2]);k3<=Math.round(max[2]);k3++){
                            myObstacle.push([k1,k2,k2]);
                        }
                function voxel2(number) {
                    var voxel=[];
                    for(var i=0;i<number;i++){
                        voxel.push([])
                        for(var j=0;j<number;j++){
                            voxel.push([])
                            for(var k=0;k<number;k++){
                                voxel.push(0)
                            }
                        }
                    }
                    for(var k1=Math.round(min[0]);k1<=Math.round(max[0]);k1++)
                        for(var k2=Math.round(min[1]);k2<=Math.round(max[1]);k2++)
                            for(var k3=Math.round(min[2]);k3<=Math.round(max[2]);k3++){
                                myObstacle.push([k1,k2,k2]);
                            }
                }
                function getPosByIndex(position,i0){//通过索引获取顶点
                    return [
                        position.array[3*i0],
                        position.array[3*i0+1],
                        position.array[3*i0+2]
                    ];
                }
            }
        }
        this.download.jsonDownload(//下载说明信息
            {
                obstacle:myObstacle

            },"grid.json"
        );
        console.log(allMax,allMin);
    },
    eigenvector:function (glb,number) {
        //var myObstacle=[];
        this.resourceManager.name=name;
        this.myGlbSplit.getArray(glb,this.resourceManager);//拆分、去除某些部件

        //var allMax;
        //var allMin;
        if(this.resourceManager.meshs>0){
            console.log("文件中包含多个mesh对象，您应当先进行拆分")
        }
        var matrix;
        var boundingBox;
        for(var k=0;k<this.resourceManager.meshs.length;k++){
            var node=this.resourceManager.meshs[k];
            node.applyMatrix(new THREE.Matrix4().identity ());//更新node.matrix
            matrix=node.matrix;
            var geometry=node.geometry;
            geometry.computeBoundingBox();
            boundingBox=geometry.boundingBox;
            var position=geometry.attributes.position;
            var arr=position.array;

            /*
            var matrix=node.matrix;
            var e=matrix.elements;
            for(var i=0;i<position.count;i++){
                var x=arr[3*i];
                var y=arr[3*i+1];
                var z=arr[3*i+2];

                arr[3*i]  =x*e[0]+y*e[4]+z*e[8]+e[12];
                arr[3*i+1]=x*e[1]+y*e[5]+z*e[9]+e[13];
                arr[3*i+2]=x*e[2]+y*e[6]+z*e[10]+e[14];
            }
            */
            for(var i=0;i<position.count;i++){
                var x=arr[3*i];
                var y=arr[3*i+1];
                var z=arr[3*i+2];

                arr[3*i]  =(x-boundingBox.min.x)/(boundingBox.max.x-boundingBox.min.x);
                arr[3*i+1]=(y-boundingBox.min.y)/(boundingBox.max.y-boundingBox.min.y);
                arr[3*i+2]=(z-boundingBox.min.z)/(boundingBox.max.z-boundingBox.min.z);
            }
            for(i=0;i<arr.length;i++){
                arr[i]*=(number-1);
            }

            var voxel=[];
            for(i1=0;i1<number;i1++){
                voxel.push([])
                for(var i2=0;i2<number;i2++){
                    voxel[i1].push([])
                    for(var i3=0;i3<number;i3++){
                        voxel[i1][i2].push(0)
                    }
                }
            }
            //console.log(voxel)
            var index=geometry.index;
            arr=index.array;
            for(i=0;i<index.count/3;i++){
                var i_p1=arr[3*i];
                var i_p2=arr[3*i+1];
                var i_p3=arr[3*i+2];
                var p1=getPosByIndex(position,i_p1);
                var p2=getPosByIndex(position,i_p2);
                var p3=getPosByIndex(position,i_p3);
                var min=[p1[0],p1[1],p1[2]];
                var max=[p1[0],p1[1],p1[2]];
                if(p2[0]<min[0])min[0]=p2[0];
                if(p2[1]<min[1])min[1]=p2[1];
                if(p2[2]<min[2])min[2]=p2[2];

                if(p2[0]>max[0])max[0]=p2[0];
                if(p2[1]>max[1])max[1]=p2[1];
                if(p2[2]>max[2])max[2]=p2[2];

                if(p3[0]<min[0])min[0]=p3[0];
                if(p3[1]<min[1])min[1]=p3[1];
                if(p3[2]<min[2])min[2]=p3[2];

                if(p3[0]>max[0])max[0]=p3[0];
                if(p3[1]>max[1])max[1]=p3[1];
                if(p3[2]>max[2])max[2]=p3[2];
                for(var k1=Math.round(min[0]);k1<=Math.round(max[0]);k1++)
                    for(var k2=Math.round(min[1]);k2<=Math.round(max[1]);k2++)
                        for(var k3=Math.round(min[2]);k3<=Math.round(max[2]);k3++){
                            voxel[k1][k2][k3]=1;
                        }
                function getPosByIndex(position,i0){//通过索引获取顶点
                    return [
                        position.array[3*i0],
                        position.array[3*i0+1],
                        position.array[3*i0+2]
                    ]
                }
            }
        }
        var vector=[]
        for(i=0;i<number;i++){
            vector.push(0);
        }
        for(i1=0;i1<number;i1++)
            for(i2=0;i2<number;i2++)
                for(i3=0;i3<number;i3++){
                    if(voxel[i1][i2][i3]===1){
                        var length=
                            Math.pow(
                                Math.pow(i1/number-0.5,2)+
                                Math.pow(i2/number-0.5,2)+
                                Math.pow(i3/number-0.5,2),
                                0.5
                            );
                        k=length/(Math.pow(3,0.5)/2);
                        k=Math.round(k*(number-1));
                        vector[k]++;
                    }
                }

        var sum=0;
        for(i=0;i<vector.length;i++)
            sum+=vector[i];
        for(i=0;i<vector.length;i++){
              vector[i]=vector[i]/sum;
        }

        glb.voxel=voxel;
        glb.eigenvector=vector;
        //console.log(glb)
        return {
                voxel:voxel,
                vector:vector,
                matrix:matrix,
                boundingBox:boundingBox
            };
    },
    eigenvector2:function (glb,number) {
        //var myObstacle=[];
        this.resourceManager.name=name;
        this.myGlbSplit.getArray(glb,this.resourceManager);//拆分、去除某些部件
        //var allMax;
        //var allMin;
        if(this.resourceManager.meshs>0){
            console.log("文件中包含多个mesh对象，您应当先进行拆分")
        }
        var matrix;
        for(var k=0;k<this.resourceManager.meshs.length;k++){
            var node=this.resourceManager.meshs[k];
            node.applyMatrix(new THREE.Matrix4().identity ());//更新node.matrix
            matrix=node.matrix;
            var geometry=node.geometry;
            geometry.computeBoundingBox();
            var boundingBox=geometry.boundingBox;
            var position=geometry.attributes.position;
            var arr=position.array;
            /*
            var matrix=node.matrix;
            var e=matrix.elements;
            for(var i=0;i<position.count;i++){
                var x=arr[3*i];
                var y=arr[3*i+1];
                var z=arr[3*i+2];
                arr[3*i]  =x*e[0]+y*e[4]+z*e[8]+e[12];
                arr[3*i+1]=x*e[1]+y*e[5]+z*e[9]+e[13];
                arr[3*i+2]=x*e[2]+y*e[6]+z*e[10]+e[14];
            }
            */
            for(var i=0;i<position.count;i++){
                var x=arr[3*i];
                var y=arr[3*i+1];
                var z=arr[3*i+2];

                arr[3*i]  =(x-boundingBox.min.x)/(boundingBox.max.x-boundingBox.min.x);
                arr[3*i+1]=(y-boundingBox.min.y)/(boundingBox.max.y-boundingBox.min.y);
                arr[3*i+2]=(z-boundingBox.min.z)/(boundingBox.max.z-boundingBox.min.z);
            }
            for(i=0;i<arr.length;i++){
                arr[i]*=(number-1);
            }

            var voxel=[];
            for(i1=0;i1<number;i1++){
                voxel.push([])
                for(var i2=0;i2<number;i2++){
                    voxel[i1].push([])
                    for(var i3=0;i3<number;i3++){
                        voxel[i1][i2].push(0)
                    }
                }
            }
            console.log(voxel)
            var index=geometry.index;
            arr=index.array;
            for(i=0;i<index.count/3;i++){
                var i_p1=arr[3*i];
                var i_p2=arr[3*i+1];
                var i_p3=arr[3*i+2];
                var p1=getPosByIndex(position,i_p1);
                var p2=getPosByIndex(position,i_p2);
                var p3=getPosByIndex(position,i_p3);
                var min=[p1[0],p1[1],p1[2]];
                var max=[p1[0],p1[1],p1[2]];

                if(p2[0]<min[0])min[0]=p2[0];
                if(p2[1]<min[1])min[1]=p2[1];
                if(p2[2]<min[2])min[2]=p2[2];

                if(p2[0]>max[0])max[0]=p2[0];
                if(p2[1]>max[1])max[1]=p2[1];
                if(p2[2]>max[2])max[2]=p2[2];

                if(p3[0]<min[0])min[0]=p3[0];
                if(p3[1]<min[1])min[1]=p3[1];
                if(p3[2]<min[2])min[2]=p3[2];

                if(p3[0]>max[0])max[0]=p3[0];
                if(p3[1]>max[1])max[1]=p3[1];
                if(p3[2]>max[2])max[2]=p3[2];

                for(var k1=Math.round(min[0]);k1<=Math.round(max[0]);k1++)
                    for(var k2=Math.round(min[1]);k2<=Math.round(max[1]);k2++)
                        for(var k3=Math.round(min[2]);k3<=Math.round(max[2]);k3++){
                            //myObstacle.push([k1,k2,k2]);
                            //console.log(k1,k2,k3)
                            voxel[k1][k2][k3]=1;
                        }

                function getPosByIndex(position,i0){//通过索引获取顶点
                    return [
                        position.array[3*i0],
                        position.array[3*i0+1],
                        position.array[3*i0+2]
                    ];
                }
            }
        }

        /*var vector=[]
        for(i=0;i<number;i++){
            vector.push(0);
        }
        for(i1=0;i1<number;i1++)
            for(i2=0;i2<number;i2++)
                for(i3=0;i3<number;i3++){
                    if(voxel[i1][i2][i3]===1){
                        var length=
                            Math.pow(
                                Math.pow(i1/number-0.5,2)+
                                Math.pow(i2/number-0.5,2)+
                                Math.pow(i3/number-0.5,2),
                                0.5
                            );
                        k=length/(Math.pow(3,0.5)/2);
                        k=Math.round(k*(number-1));
                        vector[k]++;
                    }
                }
        */

        console.log(voxel)
        var vector=[]
        for(i1=0;i1<voxel.length;i1++){
            for(i2=0;i2<voxel.length;i2++){
                for(i3=0;i3<voxel.length;i3++){
                    vector.push(voxel[i1][i2][i3])
                }
            }
        }

        var sum=0;
        for(i=0;i<vector.length;i++)
            sum+=vector[i];
        for(i=0;i<vector.length;i++){
            vector[i]=vector[i]/sum;
        }

        glb.voxel=voxel;
        glb.eigenvector=vector;
        console.log(glb)

        return {
            voxel:voxel,
            vector:vector,
            matrix:matrix
        };

    },
    variance:function (eigenvector1,eigenvector2) {
        var variance=0;
        for(var i=0;i<eigenvector1.length;i++){
            variance+=Math.pow(
                eigenvector1[i]-eigenvector2[i],
                2
            )
        }
        return variance;
    }
}
