//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4-2 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2015.12.02 at 03:26:02 PM JST 
//


package org.voltdb.compiler.deploymentfile;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for schemaType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="schemaType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="catalog"/>
 *     &lt;enumeration value="ddl"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "schemaType")
@XmlEnum
public enum SchemaType {

    @XmlEnumValue("catalog")
    CATALOG("catalog"),
    @XmlEnumValue("ddl")
    DDL("ddl");
    private final String value;

    SchemaType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static SchemaType fromValue(String v) {
        for (SchemaType c: SchemaType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
