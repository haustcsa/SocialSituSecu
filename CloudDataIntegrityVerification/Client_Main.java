package jPBC_3;

import java.util.List;

public class Client_Main {
	public static void main(String[] args) throws Exception {
		//List<String> fileMD5list = Utils.mD5Util(1, 1);//md5值文件不变则值不变
		Client_init client_init = new Client_init();
		client_init.init();
		client_init.creatTi();
		
//		List<String> fileMD5list = Utils.mD5Util(1, 1);//md5值文件不变则值不变
//		Client_init2 client_init = new Client_init2();
//		client_init.init();
//		client_init.creatTi();
		//List<String> fileMD5list = Utils.mD5Util(1, 10);
		//List<String> fileMD5list2 = Utils.mD5Utilcloud(1, 10);
		
	}
}
