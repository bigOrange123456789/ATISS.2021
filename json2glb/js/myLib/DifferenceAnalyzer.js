function DifferenceAnalyzer() {
}
DifferenceAnalyzer.prototype={
    e:{
        cylinder:[0, 0, 0, 0.01087, 0.11, 0.32, 0.3587, 0.16, 0.038, 0.0018],
        cube:[0, 0, 0, 0, 0.0553, 0.22746, 0.387, 0.2664, 0.0615, 0.00205],
        type1:[0.00329, 0.042763, 0.069, 0.1447, 0.16776, 0.25, 0.227, 0.0888, 0.006579, 0],
    },
    voxel:function(mesh,number){
        console.log("voxel")
        if(typeof (number)==="undefined")number=10;

        var geometry=mesh.geometry;
        geometry.computeBoundingBox();
        var boundingBox=geometry.boundingBox;
        var position=geometry.attributes.position;
        var arr=position.array;

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
                ];
            }
        }
        return voxel;
    },
    eigenvector:function (mesh,number) {//模型分析时用的工具
        var scope=this;
        if(typeof (number)==="undefined")number=10;
        var voxel=scope.voxel(mesh,number);
        console.log("eigenvector")
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

        return vector;
    },
    variance:function (eigenvector1,eigenvector2) {
        var scope=this;
        if(eigenvector1 instanceof THREE.Mesh)
            eigenvector1=scope.eigenvector(eigenvector1,10);

        if(scope.e[eigenvector2]){
            eigenvector2=scope.e[eigenvector2];
        }else if(eigenvector2 instanceof THREE.Mesh)
            eigenvector2=scope.eigenvector(eigenvector2,10);

        if(eigenvector1.length!==eigenvector2.length)console.log("error!")

        //console.log("variance")
        var variance=0;
        for(var i=0;i<eigenvector1.length;i++){
            variance+=Math.pow(
                eigenvector1[i]-eigenvector2[i],
                2
            )
        }
        return Math.pow(variance,0.5);
    },
    isCylinder:function (mesh) {
        var scope=this;
        var v=scope.variance(mesh,scope.cylinder);
        console.log(v);
        return v<0.065;
    },
    isCube:function (mesh) {
        var scope=this;
        var v=scope.variance(mesh,scope.cube);
        console.log(v);
        return v<0.065;
    },
    is:function (mesh,type) {
        var scope=this;
        var v=scope.variance(mesh,type);
        console.log(v);
        return v<0.03;
    },
}
