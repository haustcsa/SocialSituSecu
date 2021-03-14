package jPBC_3;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

public class Utils {
	public static List<String> mD5Util(int a,int b) {
		long startTime=System.currentTimeMillis();      
		List<String> filePaths = new ArrayList<>();
		String filePath;
		for (int i = a; i <= b; i++) {
			filePath ="F:\\userZT\\block\\"+i+".txt";
			try {
				System.out.println(md5HashCode(filePath) +":"+i+ "号文件的md5值");  
				filePaths.add(md5HashCode(filePath));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}      
		}
        long endTime=System.currentTimeMillis();
        //System.out.println("文件MD5程序运行时间： "+(endTime - startTime)+"ms");
		return filePaths;
	}
	public static List<String> mD5Utilcloud(int a,int b) {
		long startTime=System.currentTimeMillis();      
		List<String> filePaths = new ArrayList<>();
		String filePath;
		for (int i = a; i <= b; i++) {
			filePath ="F:\\userZT\\cloudBlock\\"+i+".txt";
			try {
				System.out.println(md5HashCode(filePath) +":"+i+ "号文件的md5值");  
				filePaths.add(md5HashCode(filePath));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}      
		}
        long endTime=System.currentTimeMillis();
        //System.out.println("文件MD5程序运行时间： "+(endTime - startTime)+"ms");
		return filePaths;
	}
	public static List<String> mD5Util32(int a,int b) {
		long startTime=System.currentTimeMillis();  
		List<String> filePaths32 = new ArrayList<>();
		String filePath;
		for (int i = a; i <= b; i++) {
			filePath ="F:\\file1\\"+i+".txt";
			try {
				System.out.println(md5HashCode(filePath) +":"+i+ "号文件32位的md5值"); 
				filePaths32.add(md5HashCode32(filePath));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}    
		}
		long endTime=System.currentTimeMillis();
		//System.out.println("文件32位MD5程序运行时间： "+(endTime - startTime)+"ms");
		return filePaths32;
	}
	/**
     * 获取文件的md5值 ，有可能不是32位
     * @param filePath	文件路径
     * @return
     * @throws FileNotFoundException
     */
    public static String md5HashCode(String filePath) throws FileNotFoundException{  
        FileInputStream fis = new FileInputStream(filePath);  
        return md5HashCode(fis);  
    }
    /**
     * 保证文件的MD5值为32位
     * @param filePath	文件路径
     * @return
     * @throws FileNotFoundException
     */
    public static String md5HashCode32(String filePath) throws FileNotFoundException{  
    	FileInputStream fis = new FileInputStream(filePath);  
    	return md5HashCode32(fis);  
    }
    /**
     * java获取文件的md5值  
     * @param fis 输入流
     * @return
     */
    public static String md5HashCode(InputStream fis) {   
        try {  
        	//拿到一个MD5转换器,如果想使用SHA-1或SHA-256，则传入SHA-1,SHA-256  
            MessageDigest md = MessageDigest.getInstance("MD5"); 
            
            //分多次将一个文件读入，对于大型文件而言，比较推荐这种方式，占用内存比较少。
            byte[] buffer = new byte[1024];  
            int length = -1;  
            while ((length = fis.read(buffer, 0, 1024)) != -1) {  
                md.update(buffer, 0, length);  
            }  
            fis.close();
            //转换并返回包含16个元素字节数组,返回数值范围为-128到127
  			byte[] md5Bytes  = md.digest();
            BigInteger bigInt = new BigInteger(1, md5Bytes);//1代表绝对值 
            return bigInt.toString(16);//转换为16进制
        } catch (Exception e) {  
            e.printStackTrace();  
            return "";  
        }  
    }
    /**
     * java计算文件32位md5值
     * @param fis 输入流
     * @return
     */
  	public static String md5HashCode32(InputStream fis) {
  		try {
  			//拿到一个MD5转换器,如果想使用SHA-1或SHA-256，则传入SHA-1,SHA-256  
  			MessageDigest md = MessageDigest.getInstance("MD5");
  			
  			//分多次将一个文件读入，对于大型文件而言，比较推荐这种方式，占用内存比较少。
  			byte[] buffer = new byte[1024];
  			int length = -1;
  			while ((length = fis.read(buffer, 0, 1024)) != -1) {
  				md.update(buffer, 0, length);
  			}
  			fis.close();
  			
  			//转换并返回包含16个元素字节数组,返回数值范围为-128到127
  			byte[] md5Bytes  = md.digest();
  			StringBuffer hexValue = new StringBuffer();
  			for (int i = 0; i < md5Bytes.length; i++) {
  				int val = ((int) md5Bytes[i]) & 0xff;//解释参见最下方
  				if (val < 16) {
  					/**
  					 * 如果小于16，那么val值的16进制形式必然为一位，
  					 * 因为十进制0,1...9,10,11,12,13,14,15 对应的 16进制为 0,1...9,a,b,c,d,e,f;
  					 * 此处高位补0。
  					 */
  					hexValue.append("0");
  				}
  				//这里借助了Integer类的方法实现16进制的转换 
  				hexValue.append(Integer.toHexString(val));
  			}
  			return hexValue.toString();
  		} catch (Exception e) {
  			e.printStackTrace();
  			return "";
  		}
  	}
  	public static byte[] utilFiletoByte(File file){
		byte[] byteElement=null;
		try {
			InputStream is = new FileInputStream(file);
			if (file.exists() && file.isFile()) {
				BufferedReader br = new BufferedReader(new InputStreamReader(is,"utf-8"));
				StringBuffer sb2 = new StringBuffer();
				String line = null;
				while ((line = br.readLine())!= null) {
					sb2.append(line+"\n");
				}
				br.close();
				byteElement = sb2.toString().getBytes("utf-8");
				System.out.println("读取的:"+sb2.toString());
				//System.out.println("转化的:"+element);
				return byteElement;
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		return byteElement;
	}
}
