//将glb分解为多个小文件，同时将纹理贴图单独存放为jpg格式
function GlbSplit(){
    this.names;
    this.init();
    this.resourceManager;
    this.isScreen=false;//是否有筛选
}
GlbSplit.prototype={
    init:function(){
        this.names=getNames();
        function getNames(){
            var names=[
                '室内-会议横幅（非）',
                '室内-舞台灯架（非）',
                '上层外墙',
                '窗体',
                '外部台阶与屋顶墙体',
                '网格103','网格.103_1','网格.103_2'
            ];
            arr=[22,23,25,26,27,28,29,30,31,32,33,34,35,38,39];
            for(i=0;i<arr.length;i++){
                names.push('网格0'+arr[i]);
                names.push('网格.0'+arr[i]+'_1');
                names.push('网格.0'+arr[i]+'_2');
            }
            return names;
        }
    },
    getArray:function(glb,resourceManager){
        console.log(glb)
        this.resourceManager=resourceManager;



        var arr0=[];//记录所有mesh
        glb.scene.traverse(node => {
            if (node instanceof THREE.Mesh) {
                arr0.push(node);
            }
        });
        console.log(arr0.length)

        var arr=[];//记录mesh的仿射变换
        for(var i=0;i<arr0.length;i++){
            var node=arr0[i];
            console.log(i+"/"+arr.length+":",node.name);

            node.updateMatrix();
            var node2=node.clone();
            var parent=node.parent;
            while(parent&&parent.matrix){
                console.log("parent:",parent.name);
                parent.updateMatrix();
                var m1=parent.matrix.clone();
                node2.matrix=m1.multiply(node2.matrix);
                parent=parent.parent;
            }
            var pos=new THREE.Vector3();
            var qua=new THREE.Quaternion();
            var sca=new THREE.Vector3();
            node2.matrix.decompose(pos,qua, sca);

            arr.push(node2);
            node2.position.x=pos.x;
            node2.position.y=pos.y;
            node2.position.z=pos.z;
            node2.scale.x=sca.x;
            node2.scale.y=sca.y;
            node2.scale.z=sca.z;
            node2.quaternion.x=qua.x;
            node2.quaternion.y=qua.y;
            node2.quaternion.z=qua.z;
            node2.quaternion.w=qua.w;
        }
        if(this.isScreen)
        for(i0=arr.length-1;i0>0;i0--){
            if(this.needDelete(arr[i0])){
                arr.splice(i0,1);
            }
        }

        this.resourceManager.meshs=arr;
        console.log(arr);
    },
    needDelete:function(mesh){
        for(i=0;i<this.names.length;i++){
            if(this.names[i]===mesh.name)return true;
        }
        return false;
    }

}
